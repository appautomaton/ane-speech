import Foundation

/// Rotary position tables in mlx-speech's layout: the half-width frequencies,
/// repeated twice. Positions enter the decoder only through these.
struct RoPE: Sendable {
    let headDim: Int
    private let inverseFrequencies: [Float]

    init(theta: Double, headDim: Int) {
        self.headDim = headDim
        inverseFrequencies = (0..<headDim / 2).map { 1 / pow(Float(theta), Float(2 * $0) / Float(headDim)) }
    }

    /// cos and sin for positions `position ..< position + count`, `count * headDim` values each.
    func tables(at position: Int, rows count: Int) -> (cos: [Float16], sin: [Float16]) {
        var cos: [Float16] = []
        var sin: [Float16] = []
        cos.reserveCapacity(count * headDim)
        sin.reserveCapacity(count * headDim)
        for p in position ..< position + count {
            let angles = inverseFrequencies.map { Float(p) * $0 }
            let c = angles.map { Float16(Foundation.cos($0)) }
            let s = angles.map { Float16(Foundation.sin($0)) }
            cos += c + c
            sin += s + s
        }
        return (cos, sin)
    }
}
