import Accelerate
import Foundation

/// Whisper-style log-mel features, matching mlx-speech's Qwen3-ASR extractor:
/// frames of `nFFT` reflect-padded samples every `hop`, a periodic Hann window
/// (rounded to Float32 as mlx-speech stores it), the power spectrum, the mel
/// filter bank, log10, and the last frame dropped.
///
/// `raw` gives log10 mel power. `normalize` applies the clip-wide floor of
/// (max - 8) and maps to (x + 4) / 4; streaming chooses its own floor.
public struct LogMel: Sendable {
    public let nFFT: Int
    public let hop: Int
    public let nMels: Int
    private let bins: Int
    /// [bins, nMels], as exported.
    private let filters: [Float]
    /// [nFFT, 2 * bins]: window * cos | -window * sin, so one matmul gives re | im.
    private let basis: [Double]

    public init(filters: [Float], nFFT: Int, hop: Int, nMels: Int) {
        precondition(filters.count == (nFFT / 2 + 1) * nMels)
        self.nFFT = nFFT
        self.hop = hop
        self.nMels = nMels
        self.filters = filters
        let bins = nFFT / 2 + 1
        self.bins = bins
        var basis = [Double](repeating: 0, count: nFFT * 2 * bins)
        for n in 0..<nFFT {
            let window = Double(Float(0.5 - 0.5 * cos(2 * Double.pi * Double(n) / Double(nFFT))))
            for k in 0..<bins {
                let angle = 2 * Double.pi * Double((k * n) % nFFT) / Double(nFFT)
                basis[n * 2 * bins + k] = window * cos(angle)
                basis[n * 2 * bins + bins + k] = -window * sin(angle)
            }
        }
        self.basis = basis
    }

    /// Feature frames for `samples` samples: one per hop, the last dropped.
    public func frameCount(samples: Int) -> Int { samples / hop }

    /// log10 mel power, [nMels, frames] row-major, for the whole clip.
    public func raw(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let frames = frameCount(samples: samples.count)
        let major = rawFrames(of: samples[...], from: 0, to: frames)
        var out = [Float](repeating: 0, count: nMels * frames)
        for f in 0..<frames {
            for m in 0..<nMels { out[m * frames + f] = major[f * nMels + m] }
        }
        return (out, frames)
    }

    /// log10 mel power for frames `first ..< last` of `samples`, frame-major
    /// ([frames, nMels]). These are the frames `raw` computes for the same
    /// samples; only the windows they need are read from the reflect-padded
    /// signal. A frame is final once its window ends inside the samples, so a
    /// growing stream recomputes only the frames after its last final one.
    public func rawFrames(of samples: ArraySlice<Float>, from first: Int, to last: Int) -> [Float] {
        let half = nFFT / 2
        let n = samples.count
        precondition(n > half, "need more than \(half) samples")
        guard last > first else { return [] }
        let base = samples.startIndex
        // np.pad(samples, half, mode="reflect"), read at padded index j.
        func padded(_ j: Int) -> Double {
            let i = j - half
            if i < 0 { return Double(samples[base - i]) }
            if i < n { return Double(samples[base + i]) }
            return Double(samples[base + 2 * (n - 1) - i])
        }
        let count = last - first
        var windows = [Double](repeating: 0, count: count * nFFT)
        for f in 0..<count {
            let start = (first + f) * hop
            for k in 0..<nFFT { windows[f * nFFT + k] = padded(start + k) }
        }
        var spectrum = [Double](repeating: 0, count: count * 2 * bins)
        vDSP_mmulD(windows, 1, basis, 1, &spectrum, 1, vDSP_Length(count), vDSP_Length(2 * bins), vDSP_Length(nFFT))
        var power = [Float](repeating: 0, count: count * bins)
        for f in 0..<count {
            for k in 0..<bins {
                let re = spectrum[f * 2 * bins + k]
                let im = spectrum[f * 2 * bins + bins + k]
                power[f * bins + k] = Float(re * re + im * im)
            }
        }
        var mel = [Float](repeating: 0, count: count * nMels)
        vDSP_mmul(power, 1, filters, 1, &mel, 1, vDSP_Length(count), vDSP_Length(nMels), vDSP_Length(bins))
        return mel.map { log10(max(1e-10, $0)) }
    }

    /// (max(x, floor) + 4) / 4, elementwise.
    public static func normalize(_ raw: [Float], floor: Float) -> [Float] {
        raw.map { (max($0, floor) + 4) / 4 }
    }

    /// Normalized features for a whole clip, with mlx-speech's clip-wide floor.
    public func features(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let (values, frames) = raw(samples)
        guard let peak = values.max() else { return ([], 0) }
        return (Self.normalize(values, floor: peak - 8), frames)
    }
}
