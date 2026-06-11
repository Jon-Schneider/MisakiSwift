import Foundation

final class EnglishFallbackNetwork {
  static let unknownTokenId = 3

  private let configuration: BARTConfig
  private let model: BARTFallbackModel
  private let graphemeToToken: [Character: Int]
  private let tokenToPhoneme: [Int: Character]

  private let british: Bool

  init(british: Bool, resources: MisakiResourceLoading) throws {
    configuration = try EnglishFallbackNetwork.loadConfig(from: resources)

    self.british = british

    self.model = BARTFallbackModel(
      config: configuration,
      weights: try EnglishFallbackNetwork.loadWeights(from: resources)
    )

    var graphemeDict: [Character: Int] = [:]
    for (index, grapheme) in configuration.graphemeChars.enumerated() {
      graphemeDict[grapheme] = index
    }
    self.graphemeToToken = graphemeDict

    var phonemeDict: [Int: Character] = [:]
    for (index, phoneme) in configuration.phonemeChars.enumerated() {
       phonemeDict[index] = phoneme
    }
    self.tokenToPhoneme = phonemeDict
  }

  private func graphemesToTokens(_ graphemes: String) -> [Int] {
    var tokens: [Int] = [configuration.bosTokenId]

    for char in graphemes {
      if let tokenId = graphemeToToken[char] {
        tokens.append(Int(tokenId))
      } else {
        tokens.append(EnglishFallbackNetwork.unknownTokenId)
      }
    }

    tokens.append(configuration.eosTokenId)
    return tokens
  }

  private func tokensToPhonemes(_ tokens: [Int]) -> String {
    var phonemes = ""

    for token in tokens {
      if token > EnglishFallbackNetwork.unknownTokenId {
        if let phoneme = tokenToPhoneme[Int(token)] {
          phonemes += String(phoneme)
        }
      }
    }

    return phonemes
  }

  func callAsFunction(_ word: MToken) -> (phoneme: String, rating: Int) {
    let tokenIds = graphemesToTokens(word.text)
    let generatedIds = model.generate(inputIds: tokenIds)
    let outputText = tokensToPhonemes(generatedIds)

    return (outputText, 1)
  }

  private static func loadConfig(from resources: MisakiResourceLoading) throws -> BARTConfig {
    let data = try resources.data(named: "config", withExtension: "json")
    return try JSONDecoder().decode(BARTConfig.self, from: data)
  }

  private static func loadWeights(from resources: MisakiResourceLoading) throws -> [String: SafetensorsReader.Tensor] {
    let data = try resources.data(named: "model", withExtension: "safetensors")
    return try SafetensorsReader.read(data)
  }
}
