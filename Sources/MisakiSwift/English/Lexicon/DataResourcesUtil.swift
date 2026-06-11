import Foundation

enum DataResourcesUtil {
  /// Loads the gold (high-confidence) pronunciation dictionary from the injected, accent-specific loader.
  static func loadGold(from resources: MisakiResourceLoading) throws -> [String: Any] {
    try loadDictionary(named: "gold", from: resources)
  }

  /// Loads the silver (lower-confidence) pronunciation dictionary from the injected, accent-specific loader.
  static func loadSilver(from resources: MisakiResourceLoading) throws -> [String: Any] {
    try loadDictionary(named: "silver", from: resources)
  }

  private static func loadDictionary(named name: String, from resources: MisakiResourceLoading) throws -> [String: Any] {
    let data = try resources.data(named: name, withExtension: "json")
    return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
  }
}
