public import CoreAI
public import Foundation

/// Qwen3-ForcedAligner on Core AI: times a known transcript against its audio.
///
/// Given speech and the exact text spoken, it returns a start and end for every
/// Chinese character and every English word. The model is a Qwen3-ASR whose
/// head sorts each `<timestamp>` row into one of `classes` time bins of
/// `timestamp_ms`, so one causal pass over
///
///     <|audio_start|> audio… <|audio_end|> u1 <ts><ts> u2 <ts><ts> … uN <ts><ts>
///
/// times every unit: its two timestamps are its start and end. Text units,
/// prompt, and the repair of out-of-order times follow mlx-speech's
/// `Qwen3ForcedAligner`, which follows upstream. Reads only its artifacts,
/// which hold its own encoder.
public actor Qwen3Aligner {
    public struct Config: Decodable, Sendable {
        /// The encoder's folder, relative to the checkpoint's artifacts.
        public let encoder: String
        public let sample_rate: Int
        public let n_fft: Int
        public let hop_length: Int
        public let n_mels: Int
        public let chunk_frames: Int
        public let block_frames: Int
        public let tokens_per_chunk: Int
        public let hidden_size: Int
        public let head_dim: Int
        public let rope_theta: Double
        public let audio_token_id: Int
        public let audio_start_token_id: Int
        public let audio_end_token_id: Int
        public let timestamp_token_id: Int
        public let timestamp_ms: Double
        public let classes: Int
        public let max_seconds: Double
    }

    /// One unit of the transcript and its span, in seconds.
    public struct Item: Sendable, Equatable {
        public let text: String
        public let start: Double
        public let end: Double
    }

    public enum AlignError: Error, CustomStringConvertible {
        case nothingToAlign
        case tooLong(seconds: Double, limit: Double)
        case promptTooLong(rows: Int, cache: Int)

        public var description: String {
            switch self {
            case .nothingToAlign:
                "the transcript has nothing to align: it needs letters, numbers, or CJK characters"
            case .tooLong(let seconds, let limit):
                "the audio is \(Int(seconds.rounded())) s; the aligner takes at most \(Int(limit)) s, so split it"
            case .promptTooLong(let rows, let cache):
                "audio and transcript take \(rows) rows; the model holds \(cache), so split the audio"
            }
        }
    }

    private struct EncoderMeta: Decodable { let block_frames: Int; let tokens: Int; let hidden: Int; let masked: Float }
    private struct DecoderMeta: Decodable { let segments: Int; let segment: Int; let chunk: Int; let cache: Int; let masked: Float }

    public nonisolated let config: Config
    public nonisolated let logMel: LogMel
    public nonisolated let tokenizer: ByteLevelBPE
    public let encoder: Qwen3ASREncoder
    public let decoder: Qwen3ASRDecoder
    private let embeddings: EmbeddingTable
    private let rotary: RoPE

    /// `artifacts` is the aligner's folder, such as
    /// `artifacts/qwen3_asr/qwen3_forced_aligner_0_6b`.
    public init(artifacts: URL, preferring unit: ComputeUnitKind = .neuralEngine) async throws {
        let decode = JSONDecoder()
        let runtime = artifacts.appending(path: "runtime")
        config = try decode.decode(Config.self, from: Data(contentsOf: runtime.appending(path: "runtime.json")))
        let filterData = try Data(contentsOf: runtime.appending(path: "mel_filters.f32"))
        let filters = filterData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        logMel = LogMel(filters: filters, nFFT: config.n_fft, hop: config.hop_length, nMels: config.n_mels)
        tokenizer = try ByteLevelBPE(directory: runtime.appending(path: "tokenizer"))
        embeddings = try EmbeddingTable(fp16: runtime.appending(path: "embeddings.f16"), width: config.hidden_size)

        let folder = artifacts.appending(path: config.encoder).standardized
        let enc = try decode.decode(EncoderMeta.self, from: Data(contentsOf: folder.appending(path: "encoder.json")))
        encoder = try await Qwen3ASREncoder(
            model: folder.appending(path: "encoder.aimodel"), nMels: config.n_mels,
            blockFrames: enc.block_frames, blockTokens: enc.tokens, hidden: enc.hidden,
            chunkFrames: config.chunk_frames, tokensPerChunk: config.tokens_per_chunk, masked: enc.masked,
            preferring: unit)
        let dec = try decode.decode(DecoderMeta.self, from: Data(contentsOf: artifacts.appending(path: "decoder.json")))
        decoder = try await Qwen3ASRDecoder(
            directory: artifacts, chunk: dec.chunk, cache: dec.cache, layersPerSegment: dec.segment,
            segments: dec.segments, step: false, hidden: config.hidden_size, masked: dec.masked, preferring: unit)
        rotary = RoPE(theta: config.rope_theta, headDim: config.head_dim)
    }

    /// Times each unit of `text` in 16 kHz mono `samples`. The transcript must
    /// match the speech: every unit gets a span, whether it was spoken or not.
    public func align(_ samples: [Float], text: String) async throws -> [Item] {
        let units = Self.units(text)
        guard !units.isEmpty else { throw AlignError.nothingToAlign }
        let seconds = Double(samples.count) / Double(config.sample_rate)
        guard seconds <= config.max_seconds else { throw AlignError.tooLong(seconds: seconds, limit: config.max_seconds) }
        let (mel, frames) = logMel.features(Self.normalized(samples))
        let audio = try await encoder.encode(mel: mel, frames: frames)
        let ids = promptIds(units: units, audioTokens: audio.count / config.hidden_size)
        let classes = try await timestampClasses(ids: ids, audio: audio)
        let ms = Self.repair(classes.map { Double($0) * config.timestamp_ms })
        return units.indices.map {
            Item(text: units[$0], start: Double(ms[2 * $0]) / 1000, end: Double(ms[2 * $0 + 1]) / 1000)
        }
    }

    /// The prompt: the audio span, then each unit followed by two timestamps.
    /// Added tokens split the prompt before anything else, so each unit is
    /// tokenized alone.
    public nonisolated func promptIds(units: [String], audioTokens: Int) -> [Int] {
        var ids = [config.audio_start_token_id]
        ids += [Int](repeating: config.audio_token_id, count: audioTokens)
        ids.append(config.audio_end_token_id)
        for unit in units {
            ids += tokenizer.encode(unit) + [config.timestamp_token_id, config.timestamp_token_id]
        }
        return ids
    }

    /// The class the head picks at every `<timestamp>` row of `ids`, whose audio
    /// placeholders take the rows of `audio` in order. The prompt goes through
    /// in chunks against the cache, which computes what one causal pass does;
    /// only chunks holding timestamps run the head.
    public func timestampClasses(ids: [Int], audio: [Float16]) async throws -> [Int] {
        let (hidden, chunk) = (config.hidden_size, decoder.shape.chunk)
        guard ids.count <= decoder.shape.cache else {
            throw AlignError.promptTooLong(rows: ids.count, cache: decoder.shape.cache)
        }
        await decoder.rewind(to: 0)
        var audioRow = 0
        var classes: [Int] = []
        for start in stride(from: 0, to: ids.count, by: chunk) {
            let range = start ..< min(start + chunk, ids.count)
            var rows: [Float16] = []
            rows.reserveCapacity(range.count * hidden)
            for p in range {
                if ids[p] == config.audio_token_id {
                    rows.append(contentsOf: audio[audioRow * hidden ..< (audioRow + 1) * hidden])
                    audioRow += 1
                } else {
                    embeddings.append(ids[p], to: &rows)
                }
            }
            let (cos, sin) = rotary.tables(at: start, rows: range.count)
            let stamps = range.filter { ids[$0] == config.timestamp_token_id }
            if stamps.isEmpty {
                try await decoder.feed(rows: range.count, embeddings: rows, cos: cos, sin: sin)
            } else {
                let logits = try await decoder.step(rows: range.count, embeddings: rows, cos: cos, sin: sin)
                classes += stamps.map { Qwen3ASRDecoder.argmax(logits, row: $0 - start) }
            }
        }
        return classes
    }

    // MARK: Text units

    /// Upstream's units, the same for Chinese and English: whitespace
    /// separates words; within a word only letters, numbers, and apostrophes
    /// survive, so `3.75%` becomes `375`; each CJK ideograph is a unit of its
    /// own, and the runs between them stay whole: `核心PCE` is `核`, `心`, `PCE`.
    public static func units(_ text: String) -> [String] {
        var units: [String] = []
        var word = String.UnicodeScalarView()
        func flush() {
            if !word.isEmpty { units.append(String(word)) }
            word = String.UnicodeScalarView()
        }
        for scalar in text.unicodeScalars {
            if isSpace(scalar) {
                flush()
            } else if isKept(scalar) {
                if isCJK(scalar) {
                    flush()
                    units.append(String(scalar))
                } else {
                    word.append(scalar)
                }
            }
        }
        flush()
        return units
    }

    /// Python's `str.split()` whitespace: Unicode White_Space, plus the
    /// separators U+001C ... U+001F.
    private static func isSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
    }

    private static func isKept(_ scalar: Unicode.Scalar) -> Bool {
        if scalar == "'" { return true }
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
             .decimalNumber, .letterNumber, .otherNumber:
            return true
        default:
            return false
        }
    }

    private static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF, 0x2A700...0x2B73F, 0x2B740...0x2B81F,
             0x2B820...0x2CEAF, 0xF900...0xFAFF:
            return true
        default:
            return false
        }
    }

    // MARK: Times

    /// Upstream rescales audio that exceeds ±1 by its peak, then clips.
    static func normalized(_ samples: [Float]) -> [Float] {
        let peak = samples.reduce(0) { max($0, abs($1)) }
        let scale: Float = peak > 1 ? 1 / peak : 1
        return samples.map { min(1, max(-1, $0 * scale)) }
    }

    /// Makes the times monotonic as upstream's `fix_timestamp` does. The
    /// longest non-decreasing subsequence is kept, taking the earliest
    /// predecessor and the earliest end on ties. Each run of other values is
    /// replaced: runs of one or two snap to the nearer kept neighbour, longer
    /// runs are interpolated between their kept neighbours. Results are
    /// truncated to whole milliseconds.
    static func repair(_ values: [Double]) -> [Int] {
        let n = values.count
        guard n > 0 else { return [] }
        var lengths = [Int](repeating: 1, count: n)
        var parents = [Int](repeating: -1, count: n)
        for i in 1..<n {
            var best = -1
            for j in 0..<i where values[j] <= values[i] && (best < 0 || lengths[j] > lengths[best]) { best = j }
            if best >= 0 {
                lengths[i] = lengths[best] + 1
                parents[i] = best
            }
        }
        var kept = [Bool](repeating: false, count: n)
        var index = 0
        for i in 1..<n where lengths[i] > lengths[index] { index = i }
        while index != -1 {
            kept[index] = true
            index = parents[index]
        }

        var result = values
        var i = 0
        while i < n {
            if kept[i] {
                i += 1
                continue
            }
            var j = i
            while j < n, !kept[j] { j += 1 }
            // The run i ..< j is maximal, so i - 1 and j are kept where they exist.
            let left = i > 0 ? result[i - 1] : nil
            let right = j < n ? result[j] : nil
            let count = j - i
            for k in i..<j {
                switch (left, right) {
                case (nil, let right?): result[k] = right
                case (let left?, nil): result[k] = left
                case (let left?, let right?) where count <= 2:
                    result[k] = k - (i - 1) <= j - k ? left : right
                case (let left?, let right?):
                    result[k] = left + (right - left) / Double(count + 1) * Double(k - i + 1)
                case (nil, nil): break  // unreachable: something is always kept
                }
            }
            i = j
        }
        return result.map { Int($0) }
    }
}
