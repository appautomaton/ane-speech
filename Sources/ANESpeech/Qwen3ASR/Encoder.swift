public import CoreAI
public import Foundation

/// The Qwen3-ASR audio encoder, one Core AI call per 8-second block.
///
/// The encoder attends only within blocks of `blockFrames` mel frames, so blocks
/// are independent: a clip is encoded block by block, and the last, partial
/// block is zero-padded with its padding tokens masked out.
public actor Qwen3ASREncoder {
    public let blockFrames: Int
    public let blockTokens: Int
    public let hidden: Int
    public let nMels: Int
    private let model: AIModel
    private let function: InferenceFunction
    private let masked: Float16
    private let chunkFrames: Int
    private let tokensPerChunk: Int

    public init(
        model url: URL, nMels: Int, blockFrames: Int, blockTokens: Int, hidden: Int,
        chunkFrames: Int, tokensPerChunk: Int, masked: Float, preferring unit: ComputeUnitKind = .neuralEngine
    ) async throws {
        (model, function) = try await ModelLoader.loadFunction(named: "encode", from: url, preferring: unit)
        self.nMels = nMels
        self.blockFrames = blockFrames
        self.blockTokens = blockTokens
        self.hidden = hidden
        self.chunkFrames = chunkFrames
        self.tokensPerChunk = tokensPerChunk
        self.masked = Float16(masked)
    }

    /// Encoder tokens for `frames` frames, mlx-speech's
    /// `_get_feat_extract_output_lengths`: whole chunks give `tokensPerChunk`
    /// each, and a partial chunk passes through three stride-2 convolutions.
    public nonisolated func tokens(frames: Int) -> Int {
        func halve(_ n: Int) -> Int { Int((Double(n - 1) / 2).rounded(.down)) + 1 }
        let leftover = frames % chunkFrames
        let partial = halve(halve(halve(leftover)))
        return max(0, partial + (frames / chunkFrames) * tokensPerChunk)
    }

    /// Embeddings for normalized log-mel features `mel` ([nMels, frames],
    /// row-major): `tokens(frames:) * hidden` values.
    public func encode(mel: [Float], frames: Int) async throws -> [Float16] {
        var out: [Float16] = []
        out.reserveCapacity(tokens(frames: frames) * hidden)
        for start in stride(from: 0, to: frames, by: blockFrames) {
            let count = min(blockFrames, frames - start)
            var block = [Float16](repeating: 0, count: nMels * blockFrames)
            for m in 0..<nMels {
                for f in 0..<count { block[m * blockFrames + f] = Float16(mel[m * frames + start + f]) }
            }
            out += try await encodeBlock(block, frames: count)
        }
        return out
    }

    /// Embeddings for one block: `block` is [nMels, blockFrames] normalized
    /// features whose first `count` frames are valid and the rest zero.
    /// Returns `tokens(frames: count) * hidden` values.
    public func encodeBlock(_ block: [Float16], frames count: Int) async throws -> [Float16] {
        precondition(block.count == nMels * blockFrames && count <= blockFrames)
        let valid = tokens(frames: count)
        let mask = (0..<blockTokens).map { $0 < valid ? Float16(0) : masked }
        var outputs = try await function.run(inputs: [
            "mel": NDArray(scalars: block, shape: [nMels, blockFrames]),
            "key_mask": NDArray(scalars: mask, shape: [blockTokens]),
        ])
        guard let embeddings = outputs.remove("embeddings")?.ndArray else { throw ModelLoader.LoadError.missingOutput("embeddings") }
        var out: [Float16] = []
        out.reserveCapacity(valid * hidden)
        embeddings.view(as: Float16.self).withUnsafePointer { p, _, strides in
            for t in 0..<valid {
                for c in 0..<hidden { out.append(p[t * strides[0] + c * strides[1]]) }
            }
        }
        return out
    }
}
