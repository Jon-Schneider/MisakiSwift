import Foundation
@testable import MisakiSwift

/// Builds a `MisakiDirectoryResources` over the accent-specific G2P assets the test target bundles under
/// `Resources/us` and `Resources/gb` (`model.safetensors`, `config.json`, `gold.json`, `silver.json`).
enum TestResources {
  enum ResourceError: Error {
    case missingDirectory(accent: String)
  }

  static func loader(british: Bool) throws -> MisakiDirectoryResources {
    let accent = british ? "gb" : "us"
    guard
      let configURL = Bundle.module.url(
        forResource: "config",
        withExtension: "json",
        subdirectory: "Resources/\(accent)"
      )
    else {
      throw ResourceError.missingDirectory(accent: accent)
    }
    return MisakiDirectoryResources(directory: configURL.deletingLastPathComponent())
  }
}
