import ANESpeech
import CoreAI
import Foundation

/// Times one function of a model on zero inputs: the cost of a call, apart
/// from what feeds it.
func bench(_ url: URL, functions names: [String], runs: Int) async throws {
    let model = try await ModelLoader.load(url, preferring: .neuralEngine)
    print("loaded: footprint \(footprintMB()) MB")
    for name in names.isEmpty ? Array(model.functionNames.prefix(1)) : names {
        try await bench(model, name, runs: runs)
        print("  footprint \(footprintMB()) MB")
    }
}

/// The process's physical footprint, as Activity Monitor reports memory.
func footprintMB() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? Int(info.phys_footprint) >> 20 : -1
}

private func bench(_ model: AIModel, _ name: String, runs: Int) async throws {
    guard let descriptor = model.functionDescriptor(for: name), let function = try model.loadFunction(named: name)
    else { fail("bench: no function \(name)") }
    var inputs: [String: NDArray] = [:]
    for input in descriptor.inputNames {
        guard case .ndArray(let array)? = descriptor.inputDescriptor(of: input) else { fail("bench: \(input) is not an array") }
        inputs[input] = NDArray(shape: array.shape, scalarType: array.scalarType)
    }
    for _ in 0..<3 { _ = try await function.run(inputs: inputs) }
    var times: [Duration] = []
    for _ in 0..<runs {
        let start = ContinuousClock.now
        _ = try await function.run(inputs: inputs)
        times.append(ContinuousClock.now - start)
    }
    times.sort()
    print("\(name): median \(times[runs / 2]), min \(times[0]) over \(runs) runs")
}
