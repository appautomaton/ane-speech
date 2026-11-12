// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "ane-speech",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "ANESpeech", targets: ["ANESpeech"]),
        .executable(name: "ane-speech", targets: ["ane-speech"]),
    ],
    targets: [
        .target(name: "ANESpeech"),
        .executableTarget(name: "ane-speech", dependencies: ["ANESpeech"]),
    ]
)
