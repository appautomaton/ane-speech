import Foundation

/// Confucius4-R2T2 streaming recognition: mlx-speech's `R2T2StreamSession`
/// with its incremental step, on this pipeline.
///
/// Audio is consumed in windows: the first is `chunkMs + lookaheadMs`, then
/// one every `chunkMs`. Each window decodes all audio so far, continuing from
/// the previous output minus its last `unfixedTokens` tokens, with a small
/// token budget. Text is committed append-only, by length, so a delta, once
/// returned, is never revised; `hypothesis` is the latest full guess.
///
/// Each window recomputes only what can have changed: mel frames whose window
/// reaches past the audio, encoder blocks from the first one that is not yet
/// closed or whose clipping floor moved, and decoder positions from the first
/// changed input (`Qwen3ASR.generate` decides that from the cache itself).
///
/// The prompt holds all audio so far, so a session lasts until the prompt
/// fills the decoder cache: about 16 rows per second of speech, so a minute
/// for 1024 rows. The session then stops: `full` is set, later audio is
/// dropped, and `finalize` commits the last hypothesis.
public actor R2T2Stream {
    public struct Update: Sendable {
        /// Text committed by this call.
        public let delta: String
        public let committed: String
        public let hypothesis: String
        public let language: String
        /// The decoder cache is full: the session takes no more audio.
        public let full: Bool
    }

    /// One decoding step, recorded when the session is created with
    /// `recordSteps`.
    public struct Step: Sendable {
        public let kind: String
        public let samples: Int
        public let budget: Int
        public let prefix: String
        public let promptTokens: Int
        public let audioTokens: Int
        public let generated: [Int]
        public let completion: String
        public let hypothesis: String
        public let delta: String
        public let committed: String
    }

    private static let budgetSamples = 1280
    private static let minFinalizeTokens = 16

    private let asr: Qwen3ASR
    private let forcedLanguage: String?
    private let chunkSamples: Int
    private let lookaheadSamples: Int
    private let unfixedTokens: Int
    private let context: String
    private let budgetBase: Double
    private let budgetFloor: Double
    private let finalizeCap: Int

    // Text state.
    private var awaitingFirst = true
    private var rawDecoded = ""
    private var hypothesis = ""
    private var language = ""
    private var committed = ""
    private var lastCommittedChinese = false
    private var budget: Double
    private var full = false
    public private(set) var steps: [Step]?

    // Audio: every sample fed; the first `processed` have been through a window.
    private var samples: [Float] = []
    private var processed = 0

    // Mel: raw log10 mel power, frame-major; frames before `stable` are final.
    private var raw: [Float] = []
    private var stable = 0
    private var stableMax = -Float.infinity

    // Encoder: rows for the processed audio, and the closed blocks among them
    // with the floor they were normalized with and their smallest raw value.
    private var audio: [Float16] = []
    private var blocks: [(floor: Float, rawMin: Float)] = []

    public init(
        asr: Qwen3ASR, language: String? = nil, context: String = "", chunkMs: Int = 160, lookaheadMs: Int = 160,
        unfixedTokens: Int = 1, recordSteps: Bool = false
    ) throws {
        precondition((80...2000).contains(chunkMs) && lookaheadMs >= 0 && unfixedTokens >= 0)
        if let language {
            let name = Qwen3ASR.normalizeLanguage(language)
            guard asr.config.language_suffix[name] != nil else { throw Qwen3ASR.TranscribeError.unsupportedLanguage(language) }
            forcedLanguage = name
        } else {
            forcedLanguage = nil
        }
        self.asr = asr
        let rate = Double(asr.config.sample_rate)
        chunkSamples = max(1, Int((Double(chunkMs) / 1000 * rate).rounded()))
        lookaheadSamples = Int((Double(lookaheadMs) / 1000 * rate).rounded())
        self.unfixedTokens = unfixedTokens
        self.context = context
        budget = Double(max(1, (chunkSamples + lookaheadSamples) / Self.budgetSamples))
        budgetBase = Double(max(1, chunkSamples / Self.budgetSamples))
        budgetFloor = Double(min(32, max(4, 2 * (chunkSamples / Self.budgetSamples))))
        finalizeCap = max(Self.minFinalizeTokens, Int(budget))
        steps = recordSteps ? [] : nil
    }

    // MARK: Session

    /// Appends 16 kHz mono samples and decodes every window they complete.
    public func feed(_ incoming: [Float]) async throws -> Update {
        if !full { samples += incoming }
        var deltas = String.UnicodeScalarView()
        while !full, samples.count - processed >= window {
            processed += window
            awaitingFirst = false
            let budget = Int(budget)
            let text: String, fixed: String
            do {
                (text, fixed) = try await windowStep(budget: budget)
            } catch Qwen3ASR.TranscribeError.tooLong {
                full = true
                break
            }
            let delta = lock(fixed)
            deltas.append(contentsOf: delta.unicodeScalars)
            updateBudget(delta: delta)
            hypothesis = text
            record(kind: "window", budget: budget, delta: delta)
        }
        return Update(delta: String(deltas), committed: committed, hypothesis: hypothesis, language: language, full: full)
    }

    /// Decodes the unconsumed tail with a larger budget and commits the final
    /// transcript.
    public func finalize() async throws -> (text: String, language: String) {
        guard !full, samples.count > processed else {
            _ = lock(hypothesis)
            return (committed, language)
        }
        processed = samples.count
        do {
            hypothesis = try await finishStep(budget: finalizeCap)
        } catch Qwen3ASR.TranscribeError.tooLong {
            full = true
            _ = lock(hypothesis)
            return (committed, language)
        }
        let delta = lock(hypothesis)
        record(kind: "tail", budget: finalizeCap, delta: delta)
        return (committed, language)
    }

    private var window: Int { awaitingFirst ? chunkSamples + lookaheadSamples : chunkSamples }

    // MARK: Steps

    private func windowStep(budget: Int) async throws -> (text: String, fixed: String) {
        rawDecoded = R2T2Text.head(rawDecoded, before: "|")
        let (kept, dropped) = rollback(rawDecoded, tokens: unfixedTokens, minEnd: 0)
        let prefix = R2T2Text.head(kept, before: "|")
        let completion = try await generate(prefix: prefix, budget: budget, draft: dropped)
        var raw = prefix + R2T2Text.clean(completion)

        let probed = forcedLanguage == nil ? R2T2Text.probeLanguage(raw) : nil
        if forcedLanguage == "Chinese" || probed == "Chinese" { raw = R2T2Text.removeCJKSpaces(raw) }
        let (language, text) = R2T2Text.parseStreaming(raw, language: forcedLanguage)
        if let (meta, _) = R2T2Text.split(raw, at: R2T2Text.tag) {
            raw = meta + R2T2Text.tag + text
        } else {
            raw = text
        }
        raw = R2T2Text.head(raw, before: "|")
        rawDecoded = raw

        let bodyEmpty = R2T2Text.split(raw, at: R2T2Text.tag).map { $0.1.isEmpty } ?? false
        var fixed = rollback(raw, tokens: bodyEmpty ? 0 : unfixedTokens, minEnd: 0).text
        if let (_, body) = R2T2Text.split(fixed, at: R2T2Text.tag) { fixed = body }
        fixed = R2T2Text.head(fixed, before: "|")

        guard R2T2Text.split(raw, at: R2T2Text.tag) != nil || forcedLanguage != nil else { return ("", "") }
        self.language = language
        return (R2T2Text.head(text, before: "|"), fixed)
    }

    private func finishStep(budget: Int) async throws -> String {
        // mlx-speech backs off a split multibyte character here too.
        let (kept, dropped) = rollback(rawDecoded, tokens: unfixedTokens, minEnd: 1)
        let prefix = R2T2Text.head(kept, before: "|")
        let completion = try await generate(prefix: prefix, budget: budget, draft: dropped)
        rawDecoded = R2T2Text.head(
            prefix + R2T2Text.clean(completion), before: "|")
        let (language, text) = R2T2Text.parseStreaming(rawDecoded, language: forcedLanguage)
        self.language = language
        return R2T2Text.head(text, before: "|")
    }

    /// `_detokenize_prefix`: `raw` without its last `tokens` tokens, backing
    /// off further while the cut splits a multibyte character. Also returns
    /// the tokens cut, which the model will likely write again.
    private func rollback(_ raw: String, tokens: Int, minEnd: Int) -> (text: String, dropped: [Int]) {
        let ids = asr.tokenizer.encode(raw)
        var k = tokens
        while true {
            let end = max(minEnd, ids.count - k)
            let text = end > 0 ? asr.tokenizer.decode(Array(ids.prefix(end)), skipSpecial: false) : ""
            if !text.unicodeScalars.contains("\u{FFFD}") { return (text, Array(ids.dropFirst(end))) }
            if end <= minEnd { return ("", ids) }
            k += 1
        }
    }

    /// Append-only commit by length: whatever `candidate` has past the
    /// committed length is new.
    private func lock(_ candidate: String) -> String {
        let delta = String(candidate.unicodeScalars.dropFirst(R2T2Text.count(committed)))
        committed.unicodeScalars.append(contentsOf: delta.unicodeScalars)
        return delta
    }

    /// Tokens for the next window: reset after a commit, grow while nothing
    /// commits, doubled after Chinese, capped at the floor.
    private func updateBudget(delta: String) {
        if !delta.isEmpty {
            if let chinese = R2T2Text.lastTokenIsChinese(delta) { lastCommittedChinese = chinese }
            budget = budgetBase
        } else if !lastCommittedChinese {
            budget += 0.5
        } else {
            budget = budgetBase
        }
        if lastCommittedChinese { budget *= 2 }
        budget = min(budgetFloor, budget)
    }

    // MARK: Decoding

    private var last: (prefix: String, promptTokens: Int, audioTokens: Int, generated: [Int], completion: String)?

    /// One greedy completion of the prompt continued by `prefix`, over all
    /// processed audio.
    private func generate(prefix: String, budget: Int, draft: [Int]) async throws -> String {
        let audio = try await encodeProcessed()
        let audioTokens = audio.count / asr.config.hidden_size
        let ids = try asr.promptIds(audioTokens: audioTokens, language: forcedLanguage, context: context, text: prefix)
        let tokens = try await asr.generate(ids: ids, audio: audio, maxNewTokens: budget, draft: draft)
        let completion = asr.tokenizer.decode(tokens, skipSpecial: true)
        if steps != nil { last = (prefix, ids.count, audioTokens, tokens, completion) }
        return completion
    }

    /// Encoder rows for the processed audio, recomputing only what changed.
    private func encodeProcessed() async throws -> [Float16] {
        let mel = asr.logMel
        let (hop, half, nMels) = (mel.hop, mel.nFFT / 2, mel.nMels)
        let n = processed
        let frames = mel.frameCount(samples: n)

        // A frame is final once its window ends inside the audio; only later
        // frames are recomputed. The floor is the offline one, (max - 8).
        raw.removeSubrange((stable * nMels)...)
        raw += mel.rawFrames(of: samples[..<n], from: stable, to: frames)
        let nowStable = n >= half ? min(frames, (n - half) / hop + 1) : 0
        if nowStable > stable {
            stableMax = max(stableMax, raw[(stable * nMels)..<(nowStable * nMels)].max()!)
            stable = nowStable
        }
        let tailMax = frames > stable ? raw[(stable * nMels)..<(frames * nMels)].max()! : -Float.infinity
        let floor = max(stableMax, tailMax) - 8

        // A closed block is reused while its frames are final and the floor
        // clips none of its values differently.
        let encoder = asr.encoder
        let (blockFrames, blockTokens, hidden) = (encoder.blockFrames, encoder.blockTokens, encoder.hidden)
        let chunk = asr.config.chunk_frames
        var dirty = 0
        if frames >= chunk {
            for (i, block) in blocks.enumerated() {
                if (i + 1) * blockFrames > stable { break }
                if block.floor != floor, block.rawMin < max(floor, block.floor) { break }
                dirty = i + 1
            }
        }
        blocks.removeSubrange(dirty...)
        audio.removeSubrange((dirty * blockTokens * hidden)...)
        for start in stride(from: dirty * blockFrames, to: frames, by: blockFrames) {
            let count = min(blockFrames, frames - start)
            var block = [Float16](repeating: 0, count: nMels * blockFrames)
            raw.withUnsafeBufferPointer { r in
                for f in 0..<count {
                    let row = (start + f) * nMels
                    for m in 0..<nMels { block[m * blockFrames + f] = Float16((max(r[row + m], floor) + 4) / 4) }
                }
            }
            audio += try await encoder.encodeBlock(block, frames: count)
        }
        if frames >= chunk {
            let closed = min(frames, stable) / blockFrames
            for i in stride(from: dirty, to: closed, by: 1) {
                blocks.append((floor, raw[(i * blockFrames * nMels)..<((i + 1) * blockFrames * nMels)].min()!))
            }
        }
        return audio
    }

    private func record(kind: String, budget: Int, delta: String) {
        guard steps != nil, let last else { return }
        steps!.append(Step(
            kind: kind, samples: processed, budget: budget, prefix: last.prefix, promptTokens: last.promptTokens,
            audioTokens: last.audioTokens, generated: last.generated, completion: last.completion,
            hypothesis: hypothesis, delta: delta, committed: committed))
        self.last = nil
    }
}
