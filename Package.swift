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
    .target(
      name: "MisakiSwift",
      resources: [
        // Copied under a non-reserved folder name: a static-library resource bundle is signed as a
        // shallow bundle, and codesign rejects one containing a subfolder literally named `Resources`
        // ("bundle format unrecognized"). `BundleResources` sidesteps that while keeping the subfolder.
        .copy("../../BundleResources/")
      ]
    ),
    .testTarget(
      name: "MisakiSwiftTests",
      dependencies: ["MisakiSwift"],
      resources: [
        // Word → phoneme fixtures captured from the original MLX implementation of the BART fallback
        // (running on the CPU device); FallbackParityTests asserts the Accelerate port matches them.
        .copy("Resources")
      ]
    ),
  ]
)
