public import CoreAI
public import Foundation

/// Loads `.aimodel` bundles with a compute-unit preference.
///
/// The preference is soft. Core AI places what the preferred unit supports there
/// and silently falls back to the others for the rest, so placement must be
/// confirmed by measurement, never inferred from this setting.
public enum ModelLoader {
    /// Specialized models persist in Core AI's system cache, whose location is
    /// not configurable. Entries may be purged under storage pressure or once
    /// the source model changes or is deleted; `evict` removes them now.
    static let cachePolicy = AIModelCache.Policy(
        purgeConditions: [.storagePressure, .sourceAssetChangedOrDeleted]
    )

    public static func load(
        _ url: URL,
        preferring unit: ComputeUnitKind = .neuralEngine
    ) async throws -> AIModel {
        try await AIModel.specialize(
            contentsOf: url,
            options: SpecializationOptions(preferredComputeUnitKind: unit),
            cache: .default,
            cachePolicy: cachePolicy
        )
    }

    public enum LoadError: Error {
        case missingFunction(String)
        case missingOutput(String)
    }

    /// Loads the model and one of its functions. Keep the model alive while
    /// the function is in use.
    ///
    /// A specialization cached by a different build of the same executable can
    /// make `loadFunction` throw without an error; evicting the model and
    /// specializing again recovers.
    public static func loadFunction(
        named name: String, from url: URL, preferring unit: ComputeUnitKind = .neuralEngine
    ) async throws -> (AIModel, InferenceFunction) {
        let (model, functions) = try await loadFunctions(named: [name], from: url, preferring: unit)
        return (model, functions[0])
    }

    /// Loads the model once and the named functions from it, in order.
    public static func loadFunctions(
        named names: [String], from url: URL, preferring unit: ComputeUnitKind = .neuralEngine
    ) async throws -> (AIModel, [InferenceFunction]) {
        for attempt in 0..<2 {
            let model = try await load(url, preferring: unit)
            do {
                let functions = try names.map { name in
                    guard let function = try model.loadFunction(named: name) else { throw LoadError.missingFunction(name) }
                    return function
                }
                return (model, functions)
            } catch LoadError.missingFunction(let name) {
                throw LoadError.missingFunction(name)
            } catch where attempt == 0 {
                try evict(url)
            }
        }
        preconditionFailure("unreachable: the second attempt returns or throws")
    }

    /// Removes every cached specialization of the model at `url`.
    public static func evict(_ url: URL) throws {
        try AIModelCache.default.deleteEntries(for: url)
    }
}
