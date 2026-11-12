public import CoreAI
public import Foundation

/// Qwen3-ASR (and its Confucius4-R2T2 finetune) on Core AI: log-mel front end,
/// encoder, and decoder, with the prompt and tokenizer of mlx-speech.
///
/// Reads only a checkpoint's artifacts: `decoder.json` and the decoder models,
/// and from `runtime/` the constants, the embedding table, and the tokenizer;
/// the encoder comes from the shared folder `runtime.json` names. The weights
/// the checkpoint was converted from are never read.
public actor Qwen3ASR {
    public struct Config: Decodable, Sendable {
        /// The shared encoder's folder, relative to the checkpoint's artifacts.
        public let encoder: String
        public let sample_rate: Int
        public let n_fft: Int
        public let hop_length: Int
        public let n_mels: Int
        public let chunk_frames: Int
        public let block_frames: Int
        public let tokens_per_chunk: Int
        public let hidden_size: Int
        public let head_dim: Int
        public let rope_theta: Double
        public let audio_token_id: Int
        public let eos_token_ids: [Int]
        public let prompt_prefix: [Int]
        public let prompt_suffix: [Int]
        public let language_suffix: [String: [Int]]
    }

    private struct EncoderMeta: Decodable { let block_frames: Int; let tokens: Int; let hidden: Int; let masked: Float }
    private struct DecoderMeta: Decodable { let segments: Int; let segment: Int; let chunk: Int; let cache: Int; let masked: Float; let step: Int? }

    public struct Transcript: Sendable {
        public let language: String
        public let text: String
        public let rawText: String
        public let tokens: [Int]
    }

    public enum TranscribeError: Error {
        case unsupportedLanguage(String)
        case tooLong(promptRows: Int, cache: Int)
        case audioSpan(placeholders: Int, rows: Int)
        case template
    }

    public nonisolated let config: Config
    public nonisolated let logMel: LogMel
    public nonisolated let tokenizer: ByteLevelBPE
    public let encoder: Qwen3ASREncoder
    public let decoder: Qwen3ASRDecoder
    private let embeddings: EmbeddingTable
    private let rotary: RoPE

    /// `artifacts` is a checkpoint's folder, such as
    /// `artifacts/qwen3_asr/confucius4_r2t2`.
    public init(artifacts: URL, preferring unit: ComputeUnitKind = .neuralEngine) async throws {
        let decode = JSONDecoder()
        let runtime = artifacts.appending(path: "runtime")
        config = try decode.decode(Config.self, from: Data(contentsOf: runtime.appending(path: "runtime.json")))
        let filterData = try Data(contentsOf: runtime.appending(path: "mel_filters.f32"))
        let filters = filterData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        logMel = LogMel(filters: filters, nFFT: config.n_fft, hop: config.hop_length, nMels: config.n_mels)

        tokenizer = try ByteLevelBPE(directory: runtime.appending(path: "tokenizer"))
        embeddings = try EmbeddingTable(fp16: runtime.appending(path: "embeddings.f16"), width: config.hidden_size)

        let shared = artifacts.appending(path: config.encoder).standardized
        let enc = try decode.decode(EncoderMeta.self, from: Data(contentsOf: shared.appending(path: "encoder.json")))
        encoder = try await Qwen3ASREncoder(
            model: shared.appending(path: "encoder.aimodel"), nMels: config.n_mels,
            blockFrames: enc.block_frames, blockTokens: enc.tokens, hidden: enc.hidden,
            chunkFrames: config.chunk_frames, tokensPerChunk: config.tokens_per_chunk, masked: enc.masked,
            preferring: unit)
        let dec = try decode.decode(DecoderMeta.self, from: Data(contentsOf: artifacts.appending(path: "decoder.json")))
        decoder = try await Qwen3ASRDecoder(
            directory: artifacts, chunk: dec.chunk, cache: dec.cache, layersPerSegment: dec.segment,
            segments: dec.segments, step: dec.step == 1, hidden: config.hidden_size, masked: dec.masked,
            preferring: unit)
        rotary = RoPE(theta: config.rope_theta, headDim: config.head_dim)
    }

    // MARK: Offline

    /// Transcribes 16 kHz mono samples. `language` forces the output language;
    /// nil detects it. `context` is background for the model, such as names to
    /// expect.
    public func transcribe(
        _ samples: [Float], language: String? = nil, context: String = "", maxNewTokens: Int = 448
    ) async throws -> Transcript {
        let (mel, frames) = logMel.features(samples)
        let audio = try await encoder.encode(mel: mel, frames: frames)
        let ids = try promptIds(audioTokens: audio.count / config.hidden_size, language: language, context: context)
        let tokens = try await generate(ids: ids, audio: audio, maxNewTokens: maxNewTokens)
        let raw = tokenizer.decode(tokens, skipSpecial: true).trimmingCharacters(in: .whitespacesAndNewlines)
        let (detected, text) = Self.parse(raw, forcedLanguage: language)
        return Transcript(language: detected, text: text, rawText: raw, tokens: tokens)
    }

    // MARK: Generation

    /// Prompt token ids: the template around `audioTokens` audio placeholders,
    /// with `context` as the system turn (background such as names to
    /// expect), an optional forced language, then `text`, which continues the
    /// assistant turn (R2T2 streaming feeds its committed output back here).
    ///
    /// mlx-speech tokenizes the prompt as one string. Added tokens split it,
    /// so `context` merges only with the template text around it, and `text`
    /// only with the template text after the audio span; those parts are
    /// re-tokenized together with them.
    public nonisolated func promptIds(audioTokens: Int, language: String?, context: String = "", text: String = "") throws -> [Int] {
        var prefix = config.prompt_prefix
        if !context.isEmpty {
            // "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>"
            let template = tokenizer.decode(prefix, skipSpecial: false)
            guard let end = template.range(of: "<|im_end|>") else { throw TranscribeError.template }
            prefix = tokenizer.encode(String(template[..<end.lowerBound]) + context + String(template[end.lowerBound...]))
        }
        var suffix = config.prompt_suffix
        if let language {
            guard let forced = config.language_suffix[Self.normalizeLanguage(language)]
            else { throw TranscribeError.unsupportedLanguage(language) }
            suffix += forced
        }
        if !text.isEmpty { suffix = tokenizer.encode(tokenizer.decode(suffix, skipSpecial: false) + text) }
        return prefix + [Int](repeating: config.audio_token_id, count: audioTokens) + suffix
    }

    /// Greedy decoding of the prompt `ids`, whose audio placeholders take the
    /// rows of `audio` (`hidden` values each) in order: prefill, then one token
    /// per step until an end token or `maxNewTokens`.
    ///
    /// The decoder cache keeps the longest prefix of positions whose inputs are
    /// unchanged since they were written: the same ids and, for audio
    /// placeholders, bit-identical rows. Only the rest is computed. Reuse is
    /// decided here, from what the cache actually holds, so no sequence of
    /// calls can attend to stale keys.
    ///
    /// `draft` guesses the first tokens of the answer, such as the tokens a
    /// streaming caller rolled back from its previous output. It rides along
    /// with the prompt in the same passes, and every draft token the model
    /// picks itself saves one pass. The tokens returned are the model's own
    /// greedy choices whatever the draft.
    public func generate(ids: [Int], audio: [Float16], maxNewTokens: Int, draft: [Int] = []) async throws -> [Int] {
        let hidden = config.hidden_size
        let audioId = config.audio_token_id
        let audioStart = ids.firstIndex(of: audioId) ?? ids.count
        let audioRows = audio.count / hidden
        guard audio.count == audioRows * hidden,
              ids[audioStart...].prefix(while: { $0 == audioId }).count == audioRows,
              ids.count(where: { $0 == audioId }) == audioRows
        else { throw TranscribeError.audioSpan(placeholders: ids.count(where: { $0 == audioId }), rows: audioRows) }
        let count = ids.count
        let capacity = decoder.shape.cache
        guard count > 0, count < capacity else { throw TranscribeError.tooLong(promptRows: count, cache: capacity) }
        let eos = config.eos_token_ids
        let budget = min(maxNewTokens, capacity - count)
        let fed = Array(draft.prefix(while: { !eos.contains($0) && $0 != audioId }).prefix(max(0, budget - 1)))
        let inputs = ids + fed

        // Reuse: positions are causal, so a position's keys depend only on the
        // inputs at and before it.
        let limit = min(count - 1, cached.ids.count, await decoder.filled)
        var keep = 0
        while keep < limit, ids[keep] == cached.ids[keep] {
            if ids[keep] == audioId,
               !Self.sameRow(audio, keep - audioStart, cached.audio, keep - cached.audioStart, width: hidden) { break }
            keep += 1
        }
        await decoder.rewind(to: keep)
        cached = (Array(ids[..<keep]), audio, audioStart)

        func rows(_ range: Range<Int>) -> [Float16] {
            var out: [Float16] = []
            out.reserveCapacity(range.count * hidden)
            for p in range {
                if inputs[p] == audioId {
                    let row = p - audioStart
                    out.append(contentsOf: audio[row * hidden ..< (row + 1) * hidden])
                } else {
                    embeddings.append(inputs[p], to: &out)
                }
            }
            return out
        }

        // Prefill the prompt and the draft. predicted[i] is the model's choice
        // for position count + i, read from the logits of the row before it.
        var predicted: [Int] = []
        var position = keep
        while position < inputs.count {
            let range = position ..< min(position + decoder.shape.chunk, inputs.count)
            if range.upperBound < count {
                try await feed(rows(range), at: position, rows: range.count)
            } else {
                let logits = try await step(rows(range), at: position, rows: range.count)
                for p in max(range.lowerBound, count - 1) ..< range.upperBound {
                    predicted.append(Qwen3ASRDecoder.argmax(logits, row: p - range.lowerBound))
                }
            }
            cached.ids += inputs[range]
            position = range.upperBound
        }

        var tokens: [Int] = []
        var next = predicted[0]
        var verified = 0  // draft tokens the model chose itself; their keys are cached
        var drafting = !fed.isEmpty
        position = count
        while !eos.contains(next), tokens.count < budget {
            tokens.append(next)
            if tokens.count == budget { break }
            if drafting, verified < fed.count, fed[verified] == next {
                verified += 1
                position += 1
                next = predicted[verified]
                continue
            }
            if drafting, verified < fed.count {
                // The draft diverged here: forget its later positions.
                await decoder.rewind(to: position)
                cached.ids.removeSubrange(position...)
            }
            drafting = false
            var row: [Float16] = []
            embeddings.append(next, to: &row)
            let logits = try await step(row, at: position, rows: 1)
            cached.ids.append(next)
            next = Qwen3ASRDecoder.argmax(logits, row: 0)
            position += 1
        }
        return tokens
    }

    /// What the decoder cache holds: the ids of positions `0 ..< filled`, and
    /// the audio rows its placeholders took, starting at `audioStart`.
    private var cached: (ids: [Int], audio: [Float16], audioStart: Int) = ([], [], 0)

    private static func sameRow(_ a: [Float16], _ i: Int, _ b: [Float16], _ j: Int, width: Int) -> Bool {
        guard i >= 0, j >= 0, (i + 1) * width <= a.count, (j + 1) * width <= b.count else { return false }
        return a.withUnsafeBytes { x in
            b.withUnsafeBytes { y in
                let stride = width * MemoryLayout<Float16>.stride
                return memcmp(x.baseAddress! + i * stride, y.baseAddress! + j * stride, stride) == 0
            }
        }
    }

    private func feed(_ rows: [Float16], at position: Int, rows count: Int) async throws {
        let (cos, sin) = rotary.tables(at: position, rows: count)
        try await decoder.feed(rows: count, embeddings: rows, cos: cos, sin: sin)
    }

    private func step(_ rows: [Float16], at position: Int, rows count: Int) async throws -> NDArray {
        let (cos, sin) = rotary.tables(at: position, rows: count)
        return try await decoder.step(rows: count, embeddings: rows, cos: cos, sin: sin)
    }

    // MARK: Output

    /// mlx-speech's `normalize_language_name`: first letter upper, rest lower.
    public static func normalizeLanguage(_ language: String) -> String {
        let value = language.trimmingCharacters(in: .whitespaces)
        return value.prefix(1).uppercased() + value.dropFirst().lowercased()
    }

    /// mlx-speech's `parse_asr_output`: (language, transcript) from raw output
    /// of the form "language X<asr_text>transcript".
    public static func parse(_ raw: String, forcedLanguage: String?) -> (String, String) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return ("", "") }
        if let forcedLanguage { return (normalizeLanguage(forcedLanguage), clean(text)) }
        guard let tag = text.range(of: "<asr_text>") else {
            if let match = text.firstMatch(of: /(?i)(?:^|\n+)\s*language ([A-Za-z]+)/) {
                return (normalizeLanguage(String(match.1)), clean(String(text[match.range.upperBound...])))
            }
            return ("", clean(text))
        }
        let meta = text[..<tag.lowerBound]
        let transcript = clean(String(text[tag.upperBound...]))
        if meta.lowercased().contains("language none") { return ("", transcript) }
        for line in meta.split(whereSeparator: \.isNewline) {
            let line = line.trimmingCharacters(in: .whitespaces)
            if line.lowercased().hasPrefix("language ") {
                let value = line.dropFirst("language ".count).trimmingCharacters(in: .whitespaces)
                return (value.isEmpty ? "" : normalizeLanguage(value), transcript)
            }
        }
        return ("", transcript)
    }

    /// mlx-speech's `_clean_transcript_text`: trims, and drops a generated tail
    /// such as "Human:" or "Original Answer" through the end.
    static func clean(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let tail = /(?im)^\s*(?:Original Answer|Human:?|\(bot|javascript|Copyтекст|Copy)(?s:.*)\z/
        guard let match = trimmed.firstMatch(of: tail) else { return trimmed }
        return String(trimmed[..<match.range.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
