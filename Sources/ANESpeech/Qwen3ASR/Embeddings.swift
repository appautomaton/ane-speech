public import Foundation

/// The input embedding table, `[rows, width]` Float16 in one raw little-endian
/// file that conversion exports (`runtime/embeddings.f16`). The file is
/// memory-mapped, and a lookup copies one row.
public struct EmbeddingTable: Sendable {
    public enum LoadError: Error {
        case malformed(String)
    }

    public let rows: Int
    public let width: Int
    private let data: Data

    public init(fp16 url: URL, width: Int) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        let row = width * MemoryLayout<Float16>.stride
        guard width > 0, data.count % row == 0
        else { throw LoadError.malformed("\(url.lastPathComponent): \(data.count) bytes is not rows of \(width) Float16") }
        rows = data.count / row
        self.width = width
    }

    /// Appends row `id` to `out`.
    public func append(_ id: Int, to out: inout [Float16]) {
        precondition(id >= 0 && id < rows)
        data.withUnsafeBytes { raw in
            let row = UnsafeRawBufferPointer(rebasing: raw[(id * width * 2) ..< ((id + 1) * width * 2)])
            out.append(contentsOf: row.bindMemory(to: Float16.self))
        }
    }
}
