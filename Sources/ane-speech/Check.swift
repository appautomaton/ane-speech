import ANESpeech
import Foundation

/// Checks the Swift pipeline against mlx-speech on the clips in
/// `runtime/reference.json`: features, tokenizer, and the transcript.
func check(_ asr: Qwen3ASR, artifacts: URL) async throws {
    struct Clip: Decodable {
        let wav: String
        let mel_shape: [Int]
        let text: String
        let raw_text: String
        let tokens: [Int]
    }
    let runtime = artifacts.appending(path: "runtime")
    let clips = try JSONDecoder().decode([String: Clip].self, from: Data(contentsOf: runtime.appending(path: "reference.json")))
    var failures = 0
    for (name, clip) in clips.sorted(by: { $0.key < $1.key }) {
        print("\(name):")
        let samples = try AudioFile.read(URL(filePath: clip.wav), sampleRate: Double(asr.config.sample_rate))
        let seconds = Double(samples.count) / Double(asr.config.sample_rate)

        var clock = ContinuousClock.now
        let (mel, frames) = asr.logMel.features(samples)
        let melTime = ContinuousClock.now - clock
        let expected = try Data(contentsOf: runtime.appending(path: "\(name).mel.f32"))
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let melDiff = zip(mel, expected).map { abs($0 - $1) }.max() ?? .infinity
        let melOK = frames == clip.mel_shape[1] && melDiff < 1e-3
        print("  mel: \(frames) frames in \(melTime), max |diff| vs mlx-speech \(melDiff)  \(melOK ? "ok" : "FAIL")")

        // A model may spell its output with a split BPE would not choose, so
        // encoding the text reproduces its ids only when the split was
        // canonical; otherwise the encoding must still decode to the text.
        let decoded = asr.tokenizer.decode(clip.tokens, skipSpecial: true).trimmingCharacters(in: .whitespacesAndNewlines)
        let encoded = asr.tokenizer.encode(clip.raw_text)
        let roundTrip = asr.tokenizer.decode(encoded, skipSpecial: false) == clip.raw_text
        let tokenizerOK = decoded == clip.raw_text && roundTrip
        let encodeNote = encoded == clip.tokens ? "ok" : roundTrip ? "round-trips (the model's own split differs)" : "FAIL"
        print("  tokenizer: decode \(decoded == clip.raw_text ? "ok" : "FAIL"), encode \(encodeNote)")

        clock = .now
        let audio = try await asr.encoder.encode(mel: mel, frames: frames)
        let encodeTime = ContinuousClock.now - clock
        clock = .now
        let ids = try asr.promptIds(audioTokens: audio.count / asr.config.hidden_size, language: nil)
        let tokens = try await asr.generate(ids: ids, audio: audio, maxNewTokens: 448)
        let decodeTime = ContinuousClock.now - clock
        let raw = asr.tokenizer.decode(tokens, skipSpecial: true).trimmingCharacters(in: .whitespacesAndNewlines)
        let (_, text) = Qwen3ASR.parse(raw, forcedLanguage: nil)
        let agree = zip(tokens, clip.tokens).prefix { $0 == $1 }.count
        let textOK = text == clip.text
        print("  encoder: \(audio.count / asr.config.hidden_size) tokens in \(encodeTime)")
        print("  decoder: \(ids.count) prompt rows + \(tokens.count) tokens in \(decodeTime)")
        print("  tokens: first \(agree) of \(clip.tokens.count) match mlx-speech; text \(textOK ? "identical" : "DIFFERS")")
        if !textOK { print("    swift: \(text)\n    mlx:   \(clip.text)") }
        let total = melTime + encodeTime + decodeTime
        print("  total \(total) for \(String(format: "%.1f", seconds)) s of audio")
        failures += (melOK ? 0 : 1) + (tokenizerOK ? 0 : 1) + (textOK ? 0 : 1)
    }
    if failures > 0 { fail("check: \(failures) failure(s)") }
}
