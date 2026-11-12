import ANESpeech
import Foundation

/// A resident R2T2 helper for another process: one JSON request per line on
/// stdin, one JSON reply per line on stdout.
///
///     {"op": "start", "language": "Chinese", "context": "...", "chunk_ms": 640,
///      "lookahead_ms": 160, "unfixed_tokens": 2}          -> {"ok": true}
///     {"op": "feed", "pcm": "<base64 float32 LE, 16 kHz>"} -> {"committed", "hypothesis", "language", "full"}
///     {"op": "finalize"}                                  -> {"text", "language"}
///     {"op": "transcribe", "pcm": "...", "language": "Chinese", "context": "..."}
///                                                         -> {"text", "language"}
///
/// `start`, `feed`, and `finalize` stream a take through R2T2; `transcribe`
/// decodes a whole take at once, for a checkpoint without a streaming finetune.
///
/// Any failure replies {"error": "..."}. "ANE_READY" goes to stderr once the
/// models are loaded; the helper exits when stdin closes.
func serve(_ asr: Qwen3ASR) async {
    struct Request: Decodable {
        let op: String
        let pcm: String?
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

    func samples(_ request: Request) throws -> [Float] {
        guard let pcm = request.pcm, let bytes = Data(base64Encoded: pcm)
        else { throw ServeError(description: "\(request.op) needs base64 pcm") }
        return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    FileHandle.standardError.write(Data("ANE_READY\n".utf8))
    var session: R2T2Stream?
    while let line = readLine() {
        do {
            let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
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
