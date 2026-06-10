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
      // Static linkage keeps MLX's generic Swift symbols available to release builds.
      type: .static,
      targets: ["MisakiSwift"]
    ),
  ],
  dependencies: [
    // Relaxed from exact 0.30.2 to a floor so the app can resolve a single mlx-swift alongside
    // mlx-audio-swift (Soprano narrator), which requires mlx-swift >= 0.30.6.
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
    .package(
      url: "https://github.com/Jon-Schneider/MLXUtilsLibrary.git",
      branch: "jsc/2026-06-08--static-package-product"
    )
  ],
  targets: [
    .target(
      name: "MisakiSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
     ],
     resources: [
      // Copied under a non-reserved folder name: a static-library resource bundle is signed as a
      // shallow bundle, and codesign rejects one containing a subfolder literally named `Resources`
      // ("bundle format unrecognized"). `BundleResources` sidesteps that while keeping the subfolder.
      .copy("../../BundleResources/")
     ]
    ),
    .testTarget(
      name: "MisakiSwiftTests",
      dependencies: ["MisakiSwift"]
    ),
  ]
)
