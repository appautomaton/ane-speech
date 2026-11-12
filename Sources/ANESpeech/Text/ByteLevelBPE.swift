public import Foundation

/// The byte-level BPE tokenizer of Qwen2-family models, read from a Hugging
/// Face `tokenizer.json`.
///
/// Encoding splits out added tokens, NFC-normalizes the rest, pre-tokenizes
/// with the file's split regex (matched on Unicode scalars, as Rust's regex
/// does), maps bytes to the GPT-2 byte alphabet, and applies merges by rank.
/// Decoding maps tokens back to bytes and decodes UTF-8, replacing invalid
/// sequences with U+FFFD.
public final class ByteLevelBPE: Sendable {
    public enum LoadError: Error {
        case malformed(String)
    }

    private struct Pair: Hashable, Sendable {
        let left: String
        let right: String
    }

    private struct Added: Sendable {
        let content: String
        let id: Int
        let special: Bool
    }

    private let vocab: [String: Int]
    private let strings: [Int: String]
    private let ranks: [Pair: Int]
    private let added: [Added]  // longest content first
    private let addedById: [Int: Added]
    private let byteToScalar: [Unicode.Scalar]
    private let scalarToByte: [Unicode.Scalar: UInt8]
    private nonisolated(unsafe) let split: Regex<AnyRegexOutput>

    /// Hugging Face's `Qwen2Tokenizer` split regex, for folders without a
    /// `tokenizer.json` (transformers' Qwen2 converter).
    private static let qwen2Split =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Reads a checkpoint folder's `tokenizer.json` when it has one. Otherwise
    /// reads `vocab.json`, `merges.txt`, and the added tokens of
    /// `tokenizer_config.json`, and splits as `Qwen2Tokenizer` does, which is
    /// how the official Qwen3-ASR release ships.
    public convenience init(directory: URL) throws {
        let json = directory.appending(path: "tokenizer.json")
        if FileManager.default.fileExists(atPath: json.path()) {
            try self.init(file: json)
            return
        }
        guard let vocab = try JSONSerialization.jsonObject(
                  with: Data(contentsOf: directory.appending(path: "vocab.json"))) as? [String: Int],
              let config = try JSONSerialization.jsonObject(
                  with: Data(contentsOf: directory.appending(path: "tokenizer_config.json"))) as? [String: Any],
              let decoder = config["added_tokens_decoder"] as? [String: [String: Any]]
        else { throw LoadError.malformed("vocab.json, or added_tokens_decoder in tokenizer_config.json") }
        let merges = try String(contentsOf: directory.appending(path: "merges.txt"), encoding: .utf8)
            .split(separator: "\n").filter { !$0.hasPrefix("#version") }.map(String.init)
        let added = decoder.compactMap { id, token -> Added? in
            guard let id = Int(id), let content = token["content"] as? String else { return nil }
            return Added(content: content, id: id, special: token["special"] as? Bool ?? false)
        }
        try self.init(vocab: vocab, merges: merges, added: added, split: Self.qwen2Split)
    }

    /// Reads a Hugging Face `tokenizer.json`.
    public convenience init(file: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any],
              let addedTokens = root["added_tokens"] as? [[String: Any]]
        else { throw LoadError.malformed("model, vocab, merges, or added_tokens") }
        let added = addedTokens.compactMap { token -> Added? in
            guard let content = token["content"] as? String, let id = token["id"] as? Int else { return nil }
            return Added(content: content, id: id, special: token["special"] as? Bool ?? false)
        }
        guard let pre = root["pre_tokenizer"] as? [String: Any],
              let steps = pre["pretokenizers"] as? [[String: Any]],
              let pattern = (steps.first?["pattern"] as? [String: Any])?["Regex"] as? String
        else { throw LoadError.malformed("pre_tokenizer split regex") }
        try self.init(vocab: vocab, merges: merges, added: added, split: pattern)
    }

    /// `merges` holds pairs as two-string arrays or as "left right" lines.
    private init(vocab: [String: Int], merges: [Any], added: [Added], split pattern: String) throws {
        self.vocab = vocab
        var strings = [Int: String](minimumCapacity: vocab.count)
        for (token, id) in vocab { strings[id] = token }
        self.strings = strings

        var ranks = [Pair: Int](minimumCapacity: merges.count)
        for (rank, merge) in merges.enumerated() {
            if let parts = merge as? [String], parts.count == 2 {
                ranks[Pair(left: parts[0], right: parts[1])] = rank
            } else if let line = merge as? String, let space = line.firstIndex(of: " ") {
                ranks[Pair(left: String(line[..<space]), right: String(line[line.index(after: space)...]))] = rank
            }
        }
        self.ranks = ranks

        self.added = added.sorted { $0.content.count > $1.content.count }
        addedById = Dictionary(added.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })

        // GPT-2's reversible map from bytes to printable scalars.
        var printable = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = printable.map { UInt32($0) }
        var extra: UInt32 = 0
        for byte in 0...255 where !printable.contains(byte) {
            printable.append(byte)
            scalars.append(256 + extra)
            extra += 1
        }
        var byteToScalar = [Unicode.Scalar](repeating: " ", count: 256)
        var scalarToByte = [Unicode.Scalar: UInt8]()
        for (byte, value) in zip(printable, scalars) {
            let scalar = Unicode.Scalar(value)!
            byteToScalar[byte] = scalar
            scalarToByte[scalar] = UInt8(byte)
        }
        self.byteToScalar = byteToScalar
        self.scalarToByte = scalarToByte

        split = try Regex(pattern).matchingSemantics(.unicodeScalar)
    }

    // MARK: Encoding

    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let (token, range) = firstAdded(in: rest) {
                encodeOrdinary(String(rest[..<range.lowerBound]), into: &ids)
                ids.append(token.id)
                rest = rest[range.upperBound...]
            } else {
                encodeOrdinary(String(rest), into: &ids)
                break
            }
        }
        return ids
    }

    private func firstAdded(in text: Substring) -> (Added, Range<Substring.Index>)? {
        var best: (Added, Range<Substring.Index>)?
        for token in added {
            guard let range = text.range(of: token.content) else { continue }
            if best == nil || range.lowerBound < best!.1.lowerBound {
                best = (token, range)
            }
        }
        return best
    }

    private func encodeOrdinary(_ text: String, into ids: inout [Int]) {
        guard !text.isEmpty else { return }
        let normalized = text.precomposedStringWithCanonicalMapping
        var cursor = normalized.startIndex
        for match in normalized.matches(of: split) {
            if cursor < match.range.lowerBound { encodePiece(normalized[cursor..<match.range.lowerBound], into: &ids) }
            encodePiece(normalized[match.range], into: &ids)
            cursor = match.range.upperBound
        }
        if cursor < normalized.endIndex { encodePiece(normalized[cursor...], into: &ids) }
    }

    private func encodePiece(_ piece: Substring, into ids: inout [Int]) {
        guard !piece.isEmpty else { return }
        var symbols = piece.utf8.map { String(Character(byteToScalar[Int($0)])) }
        while symbols.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0..<(symbols.count - 1) {
                if let rank = ranks[Pair(left: symbols[i], right: symbols[i + 1])], best == nil || rank < best!.rank {
                    best = (rank, i)
                }
            }
            guard let (_, at) = best else { break }
            let (left, right) = (symbols[at], symbols[at + 1])
            var merged: [String] = []
            merged.reserveCapacity(symbols.count)
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1, symbols[i] == left, symbols[i + 1] == right {
                    merged.append(left + right)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
        }
        for symbol in symbols {
            if let id = vocab[symbol] { ids.append(id) }
        }
    }

    // MARK: Decoding

    public func decode(_ ids: [Int], skipSpecial: Bool) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if let token = addedById[id] {
                if !(skipSpecial && token.special) { bytes.append(contentsOf: token.content.utf8) }
            } else if let string = strings[id] {
                for scalar in string.unicodeScalars {
                    if let byte = scalarToByte[scalar] {
                        bytes.append(byte)
                    } else {
                        bytes.append(contentsOf: String(scalar).utf8)
                    }
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    public func isSpecial(_ id: Int) -> Bool { addedById[id]?.special ?? false }
}
