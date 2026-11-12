import ANESpeech
import Foundation

/// Streams a file through an R2T2 session as if it arrived live, in pieces of
/// one window, printing committed text as it lands. Reports how long each
/// window took against its duration.
func stream(_ asr: Qwen3ASR, file: URL, language: String?, chunkMs: Int) async throws {
    let rate = asr.config.sample_rate
    let samples = try AudioFile.read(file, sampleRate: Double(rate))
    let session = try R2T2Stream(asr: asr, language: language, chunkMs: chunkMs)
    let piece = rate * chunkMs / 1000
    var latencies: [Duration] = []
    var printed = 0
    func emit(_ committed: String) {
        let fresh = String(committed.unicodeScalars.dropFirst(printed))
        printed = committed.unicodeScalars.count
        if !fresh.isEmpty {
            print(fresh, terminator: "")
            fflush(stdout)
        }
    }
    for offset in stride(from: 0, to: samples.count, by: piece) {
        let start = ContinuousClock.now
        let update = try await session.feed(Array(samples[offset..<min(offset + piece, samples.count)]))
        latencies.append(ContinuousClock.now - start)
        emit(update.committed)
    }
    let start = ContinuousClock.now
    let (text, detected) = try await session.finalize()
    let finalize = ContinuousClock.now - start
    emit(text)
    print()
    if !detected.isEmpty { print("[\(detected)]") }

    let seconds = Double(samples.count) / Double(rate)
    let total = latencies.reduce(.zero, +) + finalize
    let budget = Duration.milliseconds(chunkMs)
    let sorted = latencies.sorted()
    func percentile(_ p: Double) -> Duration { sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))] }
    let report = """
        \(String(format: "%.1f", seconds)) s of audio in \(latencies.count) pieces of \(chunkMs) ms; \
        compute \(total) (\(String(format: "%.2f", total / .seconds(seconds))) x real time)
        per piece: p50 \(percentile(0.5)), p90 \(percentile(0.9)), max \(sorted.last ?? .zero); \
        \(latencies.count(where: { $0 <= budget })) of \(latencies.count) within \(chunkMs) ms; finalize \(finalize)

        """
    FileHandle.standardError.write(Data(report.utf8))
}

/// Replays mlx-speech's R2T2 runs from every `runtime/stream_*.json`, with
/// the session settings each recorded, and compares every step: the prefix fed back, the budget, the prompt length,
/// the generated ids, and what was committed.
func streamCheck(_ asr: Qwen3ASR, artifacts: URL, only name: String? = nil) async throws {
    struct Step: Decodable {
        let kind: String
        let samples: Int
        let budget: Int
        let prefix: String
        let prompt_tokens: Int
        let audio_tokens: Int
        let generated_ids: [Int]
        let completion: String
        let delta: String
        let committed: String
    }
    struct Session: Decodable {
        let language: String?
        let context: String
        let chunk_ms: Int
        let lookahead_ms: Int
        let unfixed_token_num: Int
    }
    struct Clip: Decodable {
        let wav: String
        let session: Session?
        let samples: Int
        let steps: [Step]
        let text: String
        let language: String
    }
    let runtime = artifacts.appending(path: "runtime")
    let files = try FileManager.default.contentsOfDirectory(atPath: runtime.path())
        .filter { $0.hasPrefix("stream_") && $0.hasSuffix(".json") && (name == nil || $0 == "stream_\(name!).json") }.sorted()
    var clips: [(String, Clip)] = []
    for file in files {
        let decoded = try JSONDecoder().decode([String: Clip].self, from: Data(contentsOf: runtime.appending(path: file)))
        clips += decoded.sorted(by: { $0.key < $1.key }).map { ("\(file.dropLast(5)) \($0.key)", $0.value) }
    }
    var failures = 0
    for (name, clip) in clips {
        print("\(name):")
        let samples = try AudioFile.read(URL(filePath: clip.wav), sampleRate: Double(asr.config.sample_rate))
        guard samples.count == clip.samples else { fail("\(name): \(samples.count) samples, mlx-speech read \(clip.samples)") }
        let settings = clip.session ?? Session(language: nil, context: "", chunk_ms: 160, lookahead_ms: 160, unfixed_token_num: 1)
        let session = try R2T2Stream(
            asr: asr, language: settings.language, context: settings.context, chunkMs: settings.chunk_ms,
            lookaheadMs: settings.lookahead_ms, unfixedTokens: settings.unfixed_token_num, recordSteps: true)
        let start = ContinuousClock.now
        for offset in stride(from: 0, to: samples.count, by: 2560) {
            _ = try await session.feed(Array(samples[offset..<min(offset + 2560, samples.count)]))
        }
        let (text, language) = try await session.finalize()
        let elapsed = ContinuousClock.now - start
        let steps = await session.steps ?? []

        var sameIds = 0
        var firstDivergence: Int?
        for (i, (swift, mlx)) in zip(steps, clip.steps).enumerated() {
            let same = swift.kind == mlx.kind && swift.samples == mlx.samples && swift.budget == mlx.budget
                && swift.prefix == mlx.prefix && swift.promptTokens == mlx.prompt_tokens
                && swift.audioTokens == mlx.audio_tokens && swift.generated == mlx.generated_ids
                && (swift.delta == mlx.delta || mlx.kind == "tail")  // mlx-speech's trace leaves the tail's delta empty
            if swift.generated == mlx.generated_ids { sameIds += 1 }
            if !same, firstDivergence == nil { firstDivergence = i }
        }
        let textOK = text == clip.text && language == clip.language
        print("  steps: \(steps.count) (mlx-speech \(clip.steps.count)); generated ids identical in \(sameIds)")
        if let i = firstDivergence {
            let (s, m) = (steps[i], clip.steps[i])
            print("  first divergence at step \(i), \(s.samples) samples:")
            print("    swift: budget \(s.budget), prompt \(s.promptTokens) (\(s.audioTokens) audio), prefix \(s.prefix.debugDescription)")
            print("           ids \(s.generated) \(s.completion.debugDescription) -> delta \(s.delta.debugDescription)")
            print("    mlx:   budget \(m.budget), prompt \(m.prompt_tokens) (\(m.audio_tokens) audio), prefix \(m.prefix.debugDescription)")
            print("           ids \(m.generated_ids) \(m.completion.debugDescription) -> delta \(m.delta.debugDescription)")
        } else {
            print("  every step identical")
        }
        print("  transcript \(textOK ? "identical" : "DIFFERS"); \(elapsed) for \(String(format: "%.1f", Double(samples.count) / 16000)) s")
        if !textOK { print("    swift: [\(language)] \(text)\n    mlx:   [\(clip.language)] \(clip.text)") }
        failures += textOK ? 0 : 1
    }
    if failures > 0 { fail("stream-check: \(failures) transcript(s) differ") }
}
