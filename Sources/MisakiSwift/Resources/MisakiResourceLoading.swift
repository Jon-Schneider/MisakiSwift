import Foundation

/// Supplies the data resources MisakiSwift needs at runtime — the BART fallback checkpoint and config,
/// and the gold/silver pronunciation lexicons. The package no longer bundles these (they are ~18 MB), so
/// the host app injects a loader that reads them from wherever it installed them (e.g. a directory it
/// downloaded from Hugging Face).
///
/// Resources are looked up by a bare, accent-free name because callers inject an **accent-specific**
/// loader: an American `EnglishG2P` is given a loader over the US assets, a British one over the GB
/// assets. The four names used are `model.safetensors`, `config.json`, `gold.json`, and `silver.json`.
public protocol MisakiResourceLoading {
  /// Returns the bytes of `<name>.<ext>`, throwing if the resource is absent or unreadable.
  func data(named name: String, withExtension ext: String) throws -> Data
}

public enum MisakiResourceError: Error {
  case missingResource(name: String, ext: String)
}

/// A `MisakiResourceLoading` that reads each resource as `<directory>/<name>.<ext>`. The directory is
/// expected to hold one accent's assets (`model.safetensors`, `config.json`, `gold.json`, `silver.json`).
public struct MisakiDirectoryResources: MisakiResourceLoading {

  // MARK: Lifecycle

  public init(directory: URL) {
    self.directory = directory
  }

  // MARK: Public

  public func data(named name: String, withExtension ext: String) throws -> Data {
    let url = directory.appendingPathComponent(name).appendingPathExtension(ext)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw MisakiResourceError.missingResource(name: name, ext: ext)
    }
    return try Data(contentsOf: url)
  }

  // MARK: Private

  private let directory: URL
}
