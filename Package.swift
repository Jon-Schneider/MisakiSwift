// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MisakiSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "MisakiSwift",
      type: .static,
      targets: ["MisakiSwift"]
    ),
  ],
  targets: [
    // The library bundles no resources: the ~18 MB of G2P assets (BART checkpoint/config and the
    // gold/silver lexicons) are injected by the host via `MisakiResourceLoading`, which downloads them at
    // runtime. This keeps the assets out of the app binary.
    .target(
      name: "MisakiSwift"
    ),
    .testTarget(
      name: "MisakiSwiftTests",
      dependencies: ["MisakiSwift"],
      resources: [
        // The tests carry their own copy of the G2P assets (per accent under `Resources/us` and
        // `Resources/gb`) plus the word → phoneme fixtures captured from the original MLX implementation of
        // the BART fallback; FallbackParityTests asserts the Accelerate port matches them. These live only
        // in the test target, so they are not shipped by the library.
        .copy("Resources")
      ]
    ),
  ]
)
