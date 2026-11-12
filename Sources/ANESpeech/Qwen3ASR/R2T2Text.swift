/// mlx-speech's R2T2 streaming text rules (`models/confucius4_r2t2/streaming.py`).
///
/// Python indexes strings by code point, so every rule here works on Unicode
/// scalars, never on Swift's grapheme clusters.
enum R2T2Text {
    static let tag = "<asr_text>"

    // MARK: Characters

    /// Python's `str.isspace`: Unicode White_Space plus the ASCII separators
    /// U+001C ... U+001F.
    static func isSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace || (0x1C...0x1F).contains(s.value)
    }

    static func isCJK(_ s: Unicode.Scalar) -> Bool { (0x4E00...0x9FFF).contains(s.value) }

    private static func isASCIIAlphanumeric(_ s: Unicode.Scalar) -> Bool {
        s.isASCII && (s.properties.isAlphabetic || ("0"..."9").contains(s))
    }

    // MARK: Strings

    static func string(_ scalars: some Sequence<Unicode.Scalar>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    /// Python's `str.strip()`.
    static func strip(_ text: String) -> String {
        let s = Array(text.unicodeScalars)
        guard let first = s.firstIndex(where: { !isSpace($0) }), let last = s.lastIndex(where: { !isSpace($0) })
        else { return "" }
        return string(s[first...last])
    }

    /// `text.split(separator, 1)`: the parts before and after the first
    /// occurrence, or nil when there is none.
    static func split(_ text: String, at separator: String) -> (String, String)? {
        let s = text.unicodeScalars
        guard let range = s.firstRange(of: separator.unicodeScalars) else { return nil }
        return (String(s[..<range.lowerBound]), String(s[range.upperBound...]))
    }

    /// `text.split(separator, 1)[0]`.
    static func head(_ text: String, before separator: String) -> String {
        split(text, at: separator)?.0 ?? text
    }

    static func count(_ text: String) -> Int { text.unicodeScalars.count }

    // MARK: Rules

    private static let toChinese: [Unicode.Scalar: Unicode.Scalar] = [
        ",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：", "(": "（", ")": "）",
    ]
    private static let toEnglish = Dictionary(uniqueKeysWithValues: toChinese.map { ($1, $0) })

    /// `_normalize_punct_by_context`: punctuation takes the script of the
    /// nearest non-space character before it in the original text.
    static func normalizePunctuation(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        var previous: Unicode.Scalar?
        for s in text.unicodeScalars {
            var replaced = s
            if let p = previous, toChinese[s] != nil || toEnglish[s] != nil {
                if isCJK(p) {
                    replaced = toChinese[s] ?? s
                } else if isASCIIAlphanumeric(p) || p == "\"" || p == "'" {
                    replaced = toEnglish[s] ?? s
                }
            }
            out.append(replaced)
            if !isSpace(s) { previous = s }
        }
        return String(out)
    }

    /// A completion as the session keeps it: punctuation normalized, U+FFFD
    /// (from a token cut inside a multibyte character) removed.
    static func clean(_ completion: String) -> String {
        string(normalizePunctuation(completion).unicodeScalars.filter { $0 != "\u{FFFD}" })
    }

    /// `_CJK_SPACE_PAT.sub("", text)`: drops whitespace runs between two CJK
    /// characters.
    static func removeCJKSpaces(_ text: String) -> String {
        let s = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        while i < s.count {
            guard isSpace(s[i]) else {
                out.append(s[i])
                i += 1
                continue
            }
            var j = i
            while j < s.count, isSpace(s[j]) { j += 1 }
            if !(i > 0 && isCJK(s[i - 1]) && j < s.count && isCJK(s[j])) { out.append(contentsOf: s[i..<j]) }
            i = j
        }
        return String(out)
    }

    /// `_detect_and_fix_repetitions`: a character repeated more than
    /// `threshold` times becomes one, then the first pattern of up to 20
    /// characters repeated `threshold` times or more becomes one, recursively
    /// on the rest.
    static func fixRepetitions(_ text: String, threshold: Int = 20) -> String {
        let s = Array(text.unicodeScalars)
        var runs: [Unicode.Scalar] = []
        var i = 0
        while i < s.count {
            var n = 1
            while i + n < s.count, s[i + n] == s[i] { n += 1 }
            if n > threshold { runs.append(s[i]) } else { runs += s[i..<(i + n)] }
            i += n
        }
        return string(fixPatterns(runs[...], threshold))
    }

    private static func fixPatterns(_ v: ArraySlice<Unicode.Scalar>, _ threshold: Int, maxWidth: Int = 20) -> [Unicode.Scalar] {
        let size = v.count
        guard size >= threshold * 2 else { return Array(v) }
        let base = v.startIndex
        var out: [Unicode.Scalar] = []
        var index = 0
        while index <= size - threshold * 2 {
            for width in 1...maxWidth {
                if index + width * threshold > size { break }
                let pattern = v[(base + index)..<(base + index + width)]
                let repeated = (1..<threshold).allSatisfy { rep in
                    let start = base + index + rep * width
                    return v[start..<(start + width)].elementsEqual(pattern)
                }
                guard repeated else { continue }
                var end = index + threshold * width
                while end + width <= size, v[(base + end)..<(base + end + width)].elementsEqual(pattern) { end += width }
                return out + pattern + fixPatterns(v[(base + end)...], threshold, maxWidth: maxWidth)
            }
            out.append(v[base + index])
            index += 1
        }
        return out + v[(base + index)...]
    }

    /// The language named in the meta part before `<asr_text>`: "" for
    /// "language None" or none at all.
    private static func metaLanguage(_ meta: String) -> String {
        if meta.lowercased().contains("language none") { return "" }
        for line in meta.split(whereSeparator: \.isNewline) {
            let stripped = strip(String(line))
            guard stripped.lowercased().hasPrefix("language ") else { continue }
            let value = strip(string(stripped.unicodeScalars.dropFirst("language ".count)))
            return value.isEmpty ? "" : value.prefix(1).uppercased() + value.dropFirst().lowercased()
        }
        return ""
    }

    /// `parse_language_output(raw, None)`'s language: the probe that decides
    /// whether spaces between CJK characters go.
    static func probeLanguage(_ raw: String) -> String {
        guard let (meta, _) = split(strip(raw), at: tag) else { return "" }
        return metaLanguage(meta)
    }

    /// `parse_streaming_asr_output`: strip, fix repetitions, then split the
    /// language from the transcript at `<asr_text>`.
    static func parseStreaming(_ raw: String, language forced: String?) -> (language: String, text: String) {
        var text = strip(raw)
        guard !text.isEmpty else { return ("", "") }
        text = fixRepetitions(text)
        if let forced { return (forced, text) }
        guard let (meta, body) = split(text, at: tag) else { return ("", strip(text)) }
        return (metaLanguage(meta), strip(body))
    }

    /// Characters `split_text_to_tokens` drops: ASCII and CJK punctuation.
    private static let punctuation = Set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏".unicodeScalars)

    /// Whether the last of `split_text_to_tokens(text)` is Chinese, or nil when
    /// it has none. Tokens are Latin words and single other characters, so the
    /// last token is Chinese exactly when its last kept character is CJK.
    static func lastTokenIsChinese(_ text: String) -> Bool? {
        for s in text.unicodeScalars.reversed() where !punctuation.contains(s) && !isSpace(s) {
            return isCJK(s)
        }
        return nil
    }
}
