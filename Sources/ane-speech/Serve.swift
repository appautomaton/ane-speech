import ANESpeech
import Foundation

/// A resident helper for another process: one JSON request per line on stdin,
/// one JSON reply per line on stdout. It holds a recognizer or the aligner.
///
///     {"op": "start", "language": "Chinese", "context": "...", "chunk_ms": 640,
///      "lookahead_ms": 160, "unfixed_tokens": 2}          -> {"ok": true}
///     {"op": "feed", "pcm": "<base64 float32 LE, 16 kHz>"} -> {"committed", "hypothesis", "language", "full"}
///     {"op": "finalize"}                                  -> {"text", "language"}
///     {"op": "transcribe", "pcm": "...", "language": "Chinese", "context": "..."}
///                                                         -> {"text", "language"}
///     {"op": "align", "pcm": "...", "text": "..."}        -> {"items": [[unit, start_s, end_s], ...]}
///
/// `start`, `feed`, and `finalize` stream a take through R2T2; `transcribe`
/// decodes a whole take at once, for a checkpoint without a streaming finetune;
/// `align` times a transcript, for the aligner.
/// In place of "pcm", "path" names an audio file to read, in any format and
/// rate Core Audio reads.
///
/// Any failure replies {"error": "..."}. "ANE_READY" goes to stderr once the
/// models are loaded; the helper exits when stdin closes.
func serve(asr: Qwen3ASR? = nil, aligner: Qwen3Aligner? = nil) async {
    struct Request: Decodable {
        let op: String
        let pcm: String?
        let path: String?
        let text: String?
        let language: String?
        let context: String?
        let chunk_ms: Int?
        let lookahead_ms: Int?
        let unfixed_tokens: Int?
    }
    struct ServeError: Error, CustomStringConvertible {
        let description: String
    }
    func reply(_ object: [String: Any]) {
        var data = (try? JSONSerialization.data(withJSONObject: object, options: .withoutEscapingSlashes)) ?? Data("{}".utf8)
        data.append(0x0A)
        FileHandle.standardOutput.write(data)
    }

    let sampleRate = Double(asr?.config.sample_rate ?? aligner?.config.sample_rate ?? 16000)
    func samples(_ request: Request) throws -> [Float] {
        if let path = request.path {
            return try AudioFile.read(URL(filePath: path), sampleRate: sampleRate)
        }
        guard let pcm = request.pcm, let bytes = Data(base64Encoded: pcm)
        else { throw ServeError(description: "\(request.op) needs base64 pcm or a path") }
        return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    FileHandle.standardError.write(Data("ANE_READY\n".utf8))
    var session: R2T2Stream?
    while let line = readLine() {
        do {
            let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
            if request.op == "align" {
                guard let aligner else { throw ServeError(description: "this checkpoint does not align") }
                guard let text = request.text else { throw ServeError(description: "align needs text") }
                let items = try await aligner.align(try samples(request), text: text)
                reply(["items": items.map { [$0.text, $0.start, $0.end] as [Any] }])
                continue
            }
            guard let asr else { throw ServeError(description: "the aligner takes only align") }
            switch request.op {
            case "start":
                session = try R2T2Stream(
                    asr: asr, language: request.language, context: request.context ?? "",
                    chunkMs: request.chunk_ms ?? 160, lookaheadMs: request.lookahead_ms ?? 160,
                    unfixedTokens: request.unfixed_tokens ?? 1)
                reply(["ok": true])
            case "feed":
                guard let session else { throw ServeError(description: "feed before start") }
                let update = try await session.feed(try samples(request))
                reply(["committed": update.committed, "hypothesis": update.hypothesis,
                       "language": update.language, "full": update.full])
            case "finalize":
                guard let current = session else { throw ServeError(description: "finalize before start") }
                session = nil
                let (text, language) = try await current.finalize()
                reply(["text": text, "language": language])
            case "transcribe":
                let result = try await asr.transcribe(
                    try samples(request), language: request.language, context: request.context ?? "")
                reply(["text": result.text, "language": result.language])
            default:
                throw ServeError(description: "unknown op \(request.op)")
            }
        } catch {
            reply(["error": String(describing: error)])
        }
    }
}
