public import CoreAI
public import Foundation

/// The Qwen3-ASR text decoder: layer segments plus a vocabulary head, each a
/// Core AI model, sharing one KV cache that is allocated once and written in
/// place.
///
/// A pass feeds up to `chunk` rows. It reads the `filled` cached positions
/// through the mask, then appends its own keys and values at
/// `filled ..< filled + rows`. When the models carry a one-row `step` function,
/// a single row takes it instead: a pass costs a fixed part plus a part that
/// grows with its rows, and the wide shape would compute `chunk` rows for one. Only valid rows are written, and only after the
/// models that read the cache have finished. Positions at or past `filled` are
/// masked, so `rewind(to:)` needs to move nothing: later rows are simply
/// overwritten by the next step.
public actor Qwen3ASRDecoder {
    public struct Shape: Sendable {
        public let chunk: Int
        public let cache: Int
        /// Whether the segments have a one-row `step` function.
        public let step: Bool
        public let layersPerSegment: Int
        public let segments: Int
        public let kvHeads = 8
        public let headDim = 128
        public let hidden = 2048
    }

    public enum DecoderError: Error {
        case missingOutput(String)
        case badRows(Int)
        case cacheFull(needed: Int, capacity: Int)
    }

    public nonisolated let shape: Shape
    public private(set) var filled = 0
    /// The loaded models; an `InferenceFunction` does not keep its model alive.
    private let models: [AIModel]
    private let segments: [InferenceFunction]
    private let steps: [InferenceFunction]  // one row per pass; empty without `step`
    private let head: InferenceFunction
    private let masked: Float16
    private var keys: [NDArray]
    private var values: [NDArray]
    // Inputs for each shape, allocated once and rewritten in place by each pass,
    // and the head's input when a one-row pass feeds it.
    private var wide: Inputs
    private var narrow: Inputs
    private var headInput: NDArray

    /// One pass shape's inputs.
    private struct Inputs {
        var hidden: NDArray
        var cos: NDArray
        var sin: NDArray
        var mask: NDArray

        init(rows: Int, cache: Int, hidden width: Int, headDim: Int) {
            hidden = NDArray(shape: [rows, width], scalarType: .float16)
            cos = NDArray(shape: [rows, headDim], scalarType: .float16)
            sin = NDArray(shape: [rows, headDim], scalarType: .float16)
            mask = NDArray(shape: [rows, cache + rows], scalarType: .float16)
        }

        /// Rows past `rows` are zero and attend only to themselves, so they
        /// never reach a valid row; their keys and values are not kept.
        mutating func write(rows: Int, embeddings: [Float16], cos c: [Float16], sin s: [Float16], visible: Int, blocked: Float16) {
            Qwen3ASRDecoder.write(embeddings, rows: rows, into: &hidden)
            Qwen3ASRDecoder.write(c, rows: rows, into: &cos)
            Qwen3ASRDecoder.write(s, rows: rows, into: &sin)
            mask.mutableView(as: Float16.self).withUnsafeMutablePointer { m, shape, strides in
                let (t, l) = (shape[0], shape[1] - shape[0])
                for r in 0..<t {
                    let row = m + r * strides[0]
                    row.update(repeating: blocked, count: l + t)
                    row.update(repeating: 0, count: visible)
                    (row + l).update(repeating: 0, count: r + 1)
                }
            }
        }
    }

    /// Loads `decoder_segment<i>.aimodel`, with its `step` function when
    /// `step` is set, and `decoder_head.aimodel` from `directory`.
    public init(
        directory: URL, chunk: Int, cache: Int, layersPerSegment: Int, segments count: Int, step: Bool,
        masked: Float, preferring unit: ComputeUnitKind = .neuralEngine
    ) async throws {
        shape = Shape(chunk: chunk, cache: cache, step: step, layersPerSegment: layersPerSegment, segments: count)
        self.masked = Float16(masked)
        var models: [AIModel] = []
        var segments: [InferenceFunction] = []
        var steps: [InferenceFunction] = []
        for i in 0..<count {
            let (model, functions) = try await ModelLoader.loadFunctions(
                named: step ? ["segment", "step"] : ["segment"],
                from: directory.appending(path: "decoder_segment\(i).aimodel"), preferring: unit)
            models.append(model)
            segments.append(functions[0])
            steps += functions.dropFirst()
        }
        let (headModel, head) = try await ModelLoader.loadFunction(
            named: "head", from: directory.appending(path: "decoder_head.aimodel"), preferring: unit)
        self.models = models + [headModel]
        self.segments = segments
        self.steps = steps
        self.head = head
        let cacheShape = [layersPerSegment, 8, cache, 128]
        keys = (0..<count).map { _ in NDArray(shape: cacheShape, scalarType: .float16) }
        values = (0..<count).map { _ in NDArray(shape: cacheShape, scalarType: .float16) }
        wide = Inputs(rows: chunk, cache: cache, hidden: shape.hidden, headDim: shape.headDim)
        narrow = Inputs(rows: 1, cache: cache, hidden: shape.hidden, headDim: shape.headDim)
        headInput = NDArray(shape: [chunk, shape.hidden], scalarType: .float16)
    }

    /// Forgets every cached position from `position` on.
    public func rewind(to position: Int) {
        filled = min(filled, max(0, position))
    }

    /// Runs `rows` rows of input embeddings (`rows * hidden` values, with RoPE
    /// `cos`/`sin` of `rows * headDim` values) and returns their logits,
    /// `[chunk, vocab]`, of which the first `rows` are valid.
    public func step(rows: Int, embeddings: [Float16], cos: [Float16], sin: [Float16]) async throws -> NDArray {
        var hidden = try await run(rows: rows, embeddings: embeddings, cos: cos, sin: sin)
        if hidden.shape[0] < shape.chunk {  // a one-row pass: the head takes `chunk` rows
            Self.copyRows(hidden, into: &headInput)
            hidden = headInput
        }
        var outputs = try await head.run(inputs: ["hidden": hidden])
        return try Self.take("logits", from: &outputs)
    }

    /// `step` without the vocabulary head: fills the cache for prompt rows
    /// whose logits nobody reads.
    public func feed(rows: Int, embeddings: [Float16], cos: [Float16], sin: [Float16]) async throws {
        _ = try await run(rows: rows, embeddings: embeddings, cos: cos, sin: sin)
    }

    /// The layer segments: appends the rows' keys and values and returns the
    /// last hidden state, `[T, hidden]` for the shape that ran.
    private func run(rows: Int, embeddings: [Float16], cos: [Float16], sin: [Float16]) async throws -> NDArray {
        let one = rows == 1 && shape.step
        let (t, l, h, d) = (one ? 1 : shape.chunk, shape.cache, shape.hidden, shape.headDim)
        guard rows > 0, rows <= t, embeddings.count == rows * h, cos.count == rows * d, sin.count == rows * d
        else { throw DecoderError.badRows(rows) }
        guard filled + rows <= l else { throw DecoderError.cacheFull(needed: filled + rows, capacity: l) }

        if one {
            narrow.write(rows: rows, embeddings: embeddings, cos: cos, sin: sin, visible: filled, blocked: masked)
        } else {
            wide.write(rows: rows, embeddings: embeddings, cos: cos, sin: sin, visible: filled, blocked: masked)
        }
        let inputs = one ? narrow : wide
        let functions = one ? steps : segments
        var hidden = inputs.hidden

        for i in functions.indices {
            var outputs = try await functions[i].run(inputs: [
                "hidden": hidden, "cos": inputs.cos, "sin": inputs.sin,
                "k_cache": keys[i], "v_cache": values[i], "mask": inputs.mask,
            ])
            hidden = try Self.take("hidden_out", from: &outputs)
            let newKeys = try Self.take("k_new", from: &outputs)
            let newValues = try Self.take("v_new", from: &outputs)
            Self.append(newKeys, rows: rows, into: &keys[i], at: filled)
            Self.append(newValues, rows: rows, into: &values[i], at: filled)
        }
        filled += rows
        return hidden
    }

    /// Copies `rows` rows of `values` into the first rows of `array` [n, width]
    /// and zeroes the rest.
    private static func write(_ values: [Float16], rows: Int, into array: inout NDArray) {
        array.mutableView(as: Float16.self).withUnsafeMutablePointer { p, shape, strides in
            let width = shape[1]
            precondition(strides[1] == 1 && values.count == rows * width)
            values.withUnsafeBufferPointer { v in
                for r in 0..<shape[0] {
                    let row = p + r * strides[0]
                    if r < rows {
                        row.update(from: v.baseAddress! + r * width, count: width)
                    } else {
                        row.update(repeating: 0, count: width)
                    }
                }
            }
        }
    }

    /// Copies every row of `source` [r, width] into the first rows of `array`;
    /// the rest keep whatever they held.
    private static func copyRows(_ source: NDArray, into array: inout NDArray) {
        source.view(as: Float16.self).withUnsafePointer { s, shape, srcStrides in
            array.mutableView(as: Float16.self).withUnsafeMutablePointer { m, _, dstStrides in
                precondition(srcStrides[1] == 1 && dstStrides[1] == 1)
                for r in 0..<shape[0] { (m + r * dstStrides[0]).update(from: s + r * srcStrides[0], count: shape[1]) }
            }
        }
    }

    private static func take(_ name: String, from outputs: inout InferenceFunction.Outputs) throws -> NDArray {
        guard let array = outputs.remove(name)?.ndArray else { throw DecoderError.missingOutput(name) }
        return array
    }

    /// Copies rows `0 ..< rows` of `source` [n, kv, T, d] to rows
    /// `position ..< position + rows` of `cache` [n, kv, L, d], honouring both
    /// arrays' strides. Rows are contiguous along d in both.
    private static func append(_ source: NDArray, rows: Int, into cache: inout NDArray, at position: Int) {
        let shape = source.shape
        let (layers, heads, width) = (shape[0], shape[1], shape[3])
        source.view(as: Float16.self).withUnsafePointer { s, _, srcStrides in
            cache.mutableView(as: Float16.self).withUnsafeMutablePointer { m, _, dstStrides in
                precondition(srcStrides[3] == 1 && dstStrides[3] == 1)
                for layer in 0..<layers {
                    for head in 0..<heads {
                        for row in 0..<rows {
                            let from = layer * srcStrides[0] + head * srcStrides[1] + row * srcStrides[2]
                            let to = layer * dstStrides[0] + head * dstStrides[1] + (position + row) * dstStrides[2]
                            (m + to).update(from: s + from, count: width)
                        }
                    }
                }
            }
        }
    }

    /// Index of the largest logit in `row`.
    public static func argmax(_ logits: NDArray, row: Int) -> Int {
        let vocab = logits.shape[1]
        return logits.view(as: Float16.self).withUnsafePointer { base, _, strides in
            let p = base + row * strides[0]
            var best = 0
            for i in 1..<vocab where p[i * strides[1]] > p[best * strides[1]] { best = i }
            return best
        }
    }
}
