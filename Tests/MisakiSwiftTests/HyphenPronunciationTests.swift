import Testing
@testable import MisakiSwift

@Test func compoundWordHyphenDoesNotEmitPausePhoneme() throws {
  let englishG2P = try EnglishG2P(british: false, resources: TestResources.loader(british: false))

  let (phonemes, _) = englishG2P.phonemize(text: "The bolt-action rifle fired.")

  #expect(!phonemes.contains("—"))
}

@Test func alphanumericCompoundHyphensDoNotEmitPausePhoneme() throws {
  let englishG2P = try EnglishG2P(british: false, resources: TestResources.loader(british: false))

  let (phonemes, _) = englishG2P.phonemize(text: "A COVID-19 vaccine has a 5-year history.")

  #expect(!phonemes.contains("—"))
}

@Test func punctuationDashesStillEmitPausePhoneme() throws {
  let englishG2P = try EnglishG2P(british: false, resources: TestResources.loader(british: false))

  let (spacedHyphenPhonemes, _) = englishG2P.phonemize(text: "The bolt - however - held.")
  let (emDashPhonemes, _) = englishG2P.phonemize(text: "The bolt—however—held.")
  let (numericRangePhonemes, _) = englishG2P.phonemize(text: "Pages 10-20 are missing.")

  #expect(spacedHyphenPhonemes.filter { $0 == "—" }.count == 2)
  #expect(emDashPhonemes.filter { $0 == "—" }.count == 2)
  #expect(numericRangePhonemes.contains("—"))
}
