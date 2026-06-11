import Foundation
import Testing
@testable import MisakiSwift

/// The fixtures were captured from the original MLX implementation of `EnglishFallbackNetwork`
/// (running on the CPU device) before it was replaced with the Accelerate port. Every word must
/// phonemize identically: the fallback decides pronunciation for out-of-vocabulary words, so any
/// numeric drift in the port is audible in narration.
private func assertFallbackMatchesFixtures(british: Bool) throws {
  let fixtureName = "\(british ? "gb" : "us")_fallback_fixtures"
  let url = try #require(Bundle.module.url(forResource: fixtureName, withExtension: "json", subdirectory: "Resources"))
  let fixtures = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: String])
  #expect(!fixtures.isEmpty)

  let network = EnglishFallbackNetwork(british: british)
  for (word, expectedPhonemes) in fixtures.sorted(by: { $0.key < $1.key }) {
    let token = MToken(text: word, tokenRange: word.startIndex ..< word.endIndex, whitespace: "")
    let (phonemes, rating) = network(token)
    #expect(phonemes == expectedPhonemes, "fallback diverged from the MLX implementation for '\(word)'")
    #expect(rating == 1)
  }
}

@Test func testFallbackParity_American() async throws {
  try assertFallbackMatchesFixtures(british: false)
}

@Test func testFallbackParity_British() async throws {
  try assertFallbackMatchesFixtures(british: true)
}
