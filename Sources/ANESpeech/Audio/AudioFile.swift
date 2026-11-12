import AVFoundation
public import Foundation

/// Reads audio files as mono Float32 samples at a fixed rate.
public enum AudioFile {
    public enum ReadError: Error {
        case unsupportedFormat
        case conversion(String)
    }

    /// The file's samples, mixed to mono and resampled to `sampleRate` if needed.
    /// 16-bit PCM is scaled by 1 / 32768, as soundfile reads it.
    public static func read(_ url: URL, sampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(file.length))
        else { throw ReadError.unsupportedFormat }
        // One read can stop short of the end; keep reading until the file is consumed.
        while file.framePosition < file.length {
            let remaining = AVAudioFrameCount(file.length - file.framePosition)
            guard let part = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: remaining)
            else { throw ReadError.unsupportedFormat }
            try file.read(into: part)
            if part.frameLength == 0 { break }
            append(part, to: input)
        }
        if source.sampleRate == sampleRate, source.channelCount == 1 {
            return samples(input)
        }
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: source, to: target)
        else { throw ReadError.unsupportedFormat }
        let capacity = AVAudioFrameCount((Double(input.frameLength) * sampleRate / source.sampleRate).rounded(.up)) + 4096
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)
        else { throw ReadError.unsupportedFormat }
        let fed = Flag()
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, status in
            if fed.isSet {
                status.pointee = .endOfStream
                return nil
            }
            fed.set()
            status.pointee = .haveData
            return input
        }
        if status == .error { throw ReadError.conversion(error?.localizedDescription ?? "unknown") }
        return samples(output)
    }

    private static func append(_ part: AVAudioPCMBuffer, to buffer: AVAudioPCMBuffer) {
        let start = Int(buffer.frameLength)
        let count = Int(part.frameLength)
        for channel in 0..<Int(part.format.channelCount) {
            (buffer.floatChannelData![channel] + start).update(from: part.floatChannelData![channel], count: count)
        }
        buffer.frameLength += part.frameLength
    }

    private static func samples(_ buffer: AVAudioPCMBuffer) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }

    /// Set once by the converter's input block, which runs synchronously inside `convert`.
    private final class Flag: @unchecked Sendable {
        private(set) var isSet = false
        func set() { isSet = true }
    }
}
