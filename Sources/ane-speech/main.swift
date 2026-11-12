import ANESpeech
import CoreAI
import Foundation

let usage = """
    usage: ane-speech transcribe <audio> [--language <name>] [--context <text>]
           ane-speech stream <audio> [--language <name>] [--chunk-ms <ms>]
           ane-speech align <audio> --text <transcript>
           ane-speech serve
           ane-speech check
           ane-speech stream-check [<reference name>]
           ane-speech decoder-check
           ane-speech inspect <model.aimodel> [--prefer ane|gpu|cpu]
           ane-speech bench <model.aimodel> [--function <name>,...] [--runs <n>]
           ane-speech evict <model.aimodel>

    options: --checkpoint <name> confucius4_r2t2 (default), qwen3_asr_1_7b,
                                 or qwen3_forced_aligner_0_6b (the default for align)
             --artifacts <dir>   a checkpoint's artifacts (default: artifacts/qwen3_asr/<checkpoint>)
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

func describe(_ value: InferenceValue.Descriptor?) -> String {
    guard let value else { return "?" }
    switch value {
    case .ndArray(let array): return "\(array.scalarType) \(array.shape)"
    case .image(let image): return "image \(image)"
    @unknown default: return "unknown"
    }
}

var args = Array(CommandLine.arguments.dropFirst())

/// Removes `--name value` from `args` and returns the value.
@MainActor func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name) else { return nil }
    guard i + 1 < args.count else { fail(usage) }
    let value = args[i + 1]
    args.removeSubrange(i...(i + 1))
    return value
}

let checkpointOption = option("--checkpoint")
let artifactsOption = option("--artifacts")
guard let command = args.first else { fail(usage) }
let checkpoint = checkpointOption ?? (command == "align" ? "qwen3_forced_aligner_0_6b" : "confucius4_r2t2")
let artifacts = URL(filePath: artifactsOption ?? "artifacts/qwen3_asr/\(checkpoint)")

func loadASR() async throws -> Qwen3ASR {
    let clock = ContinuousClock.now
    let asr = try await Qwen3ASR(artifacts: artifacts)
    FileHandle.standardError.write(Data("loaded in \(ContinuousClock.now - clock)\n".utf8))
    return asr
}

func loadAligner() async throws -> Qwen3Aligner {
    let clock = ContinuousClock.now
    let aligner = try await Qwen3Aligner(artifacts: artifacts)
    FileHandle.standardError.write(Data("loaded in \(ContinuousClock.now - clock)\n".utf8))
    return aligner
}

/// Whether `artifacts` holds the aligner: its runtime.json names a timestamp token.
func holdsAligner() throws -> Bool {
    struct Probe: Decodable { let timestamp_token_id: Int? }
    let url = artifacts.appending(path: "runtime/runtime.json")
    return try JSONDecoder().decode(Probe.self, from: Data(contentsOf: url)).timestamp_token_id != nil
}

switch command {
case "transcribe":
    let language = option("--language")
    let context = option("--context") ?? ""
    guard args.count == 2 else { fail(usage) }
    let asr = try await loadASR()
    let samples = try AudioFile.read(URL(filePath: args[1]), sampleRate: Double(asr.config.sample_rate))
    let result = try await asr.transcribe(samples, language: language, context: context)
    print(result.language.isEmpty ? result.text : "[\(result.language)] \(result.text)")
case "stream":
    let language = option("--language")
    guard let chunkMs = Int(option("--chunk-ms") ?? "160"), args.count == 2 else { fail(usage) }
    try await stream(try await loadASR(), file: URL(filePath: args[1]), language: language, chunkMs: chunkMs)
case "align":
    guard let text = option("--text"), args.count == 2 else { fail(usage) }
    let aligner = try await loadAligner()
    let samples = try AudioFile.read(URL(filePath: args[1]), sampleRate: Double(aligner.config.sample_rate))
    for item in try await aligner.align(samples, text: text) {
        print(String(format: "%.3f\t%.3f\t", item.start, item.end) + item.text)
    }
case "serve":
    do {
        if try holdsAligner() {
            await serve(aligner: try await loadAligner())
        } else {
            await serve(asr: try await loadASR())
        }
    } catch {
        fail("ANE_ERROR: \(error)")
    }
case "check":
    if try holdsAligner() {
        try await alignCheck(try await loadAligner(), artifacts: artifacts)
    } else {
        try await check(try await loadASR(), artifacts: artifacts)
    }
case "stream-check":
    try await streamCheck(try await loadASR(), artifacts: artifacts, only: args.count > 1 ? args[1] : nil)
case "decoder-check":
    try await decoderCheck(artifacts)
case "inspect":
    var unit = ComputeUnitKind.neuralEngine
    switch option("--prefer") {
    case nil, "ane": unit = .neuralEngine
    case "gpu": unit = .gpu
    case "cpu": unit = .cpu
    default: fail(usage)
    }
    guard args.count == 2 else { fail(usage) }
    let model = try await ModelLoader.load(URL(filePath: args[1]), preferring: unit)
    for name in model.functionNames {
        guard let function = model.functionDescriptor(for: name) else { continue }
        print(name)
        for input in function.inputNames {
            print("  in    \(input): \(describe(function.inputDescriptor(of: input)))")
        }
        for state in function.stateNames {
            print("  state \(state): \(describe(function.stateDescriptor(of: state)))")
        }
        for output in function.outputNames {
            print("  out   \(output): \(describe(function.outputDescriptor(of: output)))")
        }
    }
case "bench":
    let functions = option("--function")?.split(separator: ",").map(String.init) ?? []
    guard let runs = Int(option("--runs") ?? "20"), args.count == 2 else { fail(usage) }
    try await bench(URL(filePath: args[1]), functions: functions, runs: runs)
case "evict":
    guard args.count == 2 else { fail(usage) }
    try ModelLoader.evict(URL(filePath: args[1]))
default:
    fail(usage)
}
