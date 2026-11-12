import ANESpeech
import CoreAI
import Foundation

/// Teacher-forced decode of the reference clip: prefill in chunks, then one
/// token per step, comparing each greedy token with mlx-speech's.
func decoderCheck(_ directory: URL) async throws {
    struct Meta: Decodable {
        let rows: Int
        let prompt_rows: Int
        let tokens: [Int]
        let segment: Int
        let chunk: Int
        let cache: Int
        let masked: Float
    }
    struct Shapes: Decodable { let step: Int? }
    let oneRow = try JSONDecoder().decode(Shapes.self, from: Data(contentsOf: directory.appending(path: "decoder.json"))).step == 1
    let refDir = directory.appending(path: "decoder_reference")
    let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: refDir.appending(path: "meta.json")))
    func halves(_ name: String) throws -> [Float16] {
        let data = try Data(contentsOf: refDir.appending(path: name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }
    let (embeds, cos, sin) = (try halves("embeds.f16"), try halves("cos.f16"), try halves("sin.f16"))
    let segments = try FileManager.default.contentsOfDirectory(atPath: directory.path(percentEncoded: false))
        .filter { $0.hasPrefix("decoder_segment") && $0.hasSuffix(".aimodel") }.count

    var clock = ContinuousClock.now
    let decoder = try await Qwen3ASRDecoder(
        directory: directory, chunk: meta.chunk, cache: meta.cache,
        layersPerSegment: meta.segment, segments: segments, step: oneRow, masked: meta.masked)
    print("loaded \(segments) segments and the head in \(ContinuousClock.now - clock)")
    let (h, d) = (decoder.shape.hidden, decoder.shape.headDim)

    func step(_ rows: Range<Int>) async throws -> NDArray {
        try await decoder.step(
            rows: rows.count,
            embeddings: Array(embeds[rows.lowerBound * h ..< rows.upperBound * h]),
            cos: Array(cos[rows.lowerBound * d ..< rows.upperBound * d]),
            sin: Array(sin[rows.lowerBound * d ..< rows.upperBound * d]))
    }

    var predicted: [Int] = []
    clock = .now
    var row = 0
    while row < meta.prompt_rows {
        let rows = row ..< min(row + meta.chunk, meta.prompt_rows)
        let logits = try await step(rows)
        row = rows.upperBound
        if row == meta.prompt_rows { predicted.append(Qwen3ASRDecoder.argmax(logits, row: rows.count - 1)) }
    }
    let prefill = ContinuousClock.now - clock
    clock = .now
    for r in meta.prompt_rows ..< meta.rows {
        predicted.append(Qwen3ASRDecoder.argmax(try await step(r ..< r + 1), row: 0))
    }
    let decode = ContinuousClock.now - clock
    let steps = meta.rows - meta.prompt_rows
    let agree = zip(predicted, meta.tokens).filter { $0 == $1 }.count
    print("prefill \(meta.prompt_rows) rows: \(prefill)")
    print("decode \(steps) tokens: \(decode / steps) per token")
    print("greedy tokens agree with mlx-speech: \(agree)/\(meta.tokens.count)")
    if agree != meta.tokens.count { fail("decoder check failed: \(predicted) vs \(meta.tokens)") }
}
