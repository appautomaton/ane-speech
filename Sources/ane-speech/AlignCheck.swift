import ANESpeech
import Foundation

/// Checks the aligner against mlx-speech on the clips in
/// `runtime/reference.json`: text units and prompt ids exactly, time classes
/// to within one class at near-ties, and the spans they make.
func alignCheck(_ aligner: Qwen3Aligner, artifacts: URL) async throws {
    struct Span: Decodable {
        let start: Double
        let end: Double

        init(from decoder: any Decoder) throws {  // [text, start, end]
            var fields = try decoder.unkeyedContainer()
            _ = try fields.decode(String.self)
            start = try fields.decode(Double.self)
            end = try fields.decode(Double.self)
        }
    }
    struct Clip: Decodable {
        let wav: String
        let text: String
        let units: [String]
        let input_ids: [Int]
        let classes: [Int]
        let spans: [Span]
    }
    let runtime = artifacts.appending(path: "runtime")
    let clips = try JSONDecoder().decode([String: Clip].self, from: Data(contentsOf: runtime.appending(path: "reference.json")))
    var failures = 0
    for (name, clip) in clips.sorted(by: { $0.key < $1.key }) {
        print("\(name):")
        let samples = try AudioFile.read(URL(filePath: clip.wav), sampleRate: Double(aligner.config.sample_rate))
        let seconds = Double(samples.count) / Double(aligner.config.sample_rate)

        let units = Qwen3Aligner.units(clip.text)
        let (mel, frames) = aligner.logMel.features(samples)
        let audio = try await aligner.encoder.encode(mel: mel, frames: frames)
        let ids = aligner.promptIds(units: units, audioTokens: audio.count / aligner.config.hidden_size)
        let promptOK = units == clip.units && ids == clip.input_ids
        print("  units \(units == clip.units ? "identical" : "DIFFER") (\(units.count)), "
              + "prompt ids \(ids == clip.input_ids ? "identical" : "DIFFER") (\(ids.count))")

        let classes = try await aligner.timestampClasses(ids: ids, audio: audio)
        let off = zip(classes, clip.classes).map { abs($0 - $1) }
        let agree = Double(off.count(where: { $0 == 0 })) / Double(max(off.count, 1))
        let classesOK = classes.count == clip.classes.count && (off.max() ?? 0) <= 1 && agree >= 0.98
        print("  time classes: \(String(format: "%.1f", agree * 100))% match mlx-speech, "
              + "off by at most \(off.max() ?? 0)  \(classesOK ? "ok" : "FAIL")")

        let clock = ContinuousClock.now
        let items = try await aligner.align(samples, text: clip.text)
        let took = ContinuousClock.now - clock
        let spanDiff = zip(items, clip.spans).map { max(abs($0.start - $1.start), abs($0.end - $1.end)) }.max() ?? 0
        print("  spans: max |diff| vs mlx-speech \(Int((spanDiff * 1000).rounded())) ms")
        print("  aligned \(items.count) units in \(took) for \(String(format: "%.1f", seconds)) s of audio")
        failures += (promptOK ? 0 : 1) + (classesOK ? 0 : 1)
    }
    if failures > 0 { fail("check: \(failures) failure(s)") }
}
