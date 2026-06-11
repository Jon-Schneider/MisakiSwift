# MisakiSwift

A Swift port of the [Misaki](https://github.com/hexgrad/misaki) grapheme-to-phoneme (G2P) library for converting English text to phonetic representations suitable for text-to-speech (TTS) engines.

## Supported Platforms

- iOS 18.0+
- macOS 15.0+
- (Other Apple platforms may work as well)

## Installation

Add MisakiSwift to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/mlalma/MisakiSwift", from: "1.0.1")
]
```

## Basic Usage

```swift
import MisakiSwift

// Create G2P converter (british = false for American English)
let g2p = EnglishG2P(british: false)

// Convert text to phonemes
let (phonemes, tokens) = g2p.phonemize(text: "Hello world!")
print(phonemes) // "həlˈO wˈɜɹld!"
```

## Custom Phoneme Override

Use Markdown-like syntax to specify exact pronunciations in case you don't want to use fallback network:

```swift
let g2p = EnglishG2P(british: false)
let text = "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models."
let (phonemes, _) = g2p.phonemize(text: text)
// "misˈɑki ɪz ɐ ʤˈitəpˈi ˈɛnʤən dəzˈInd fɔɹ kˈOkəɹO mˈɑdᵊlz."
```

## Overview

MisakiSwift is a high-quality English G2P conversion library that transforms written text into phonemes using both dictionary-based lookup and neural network fallback. It supports British and American English pronunciations and includes advanced features like stress pattern handling and custom phoneme overrides.

## Key Features

- **High Accuracy**: Combines extensive pronunciation dictionaries with neural network fallback for out-of-vocabulary words
- **Dual Dialect Support**: Supports both British and American English pronunciations
- **Advanced Text Processing**: Handles punctuation, numbers, acronyms, and complex formatting
- **Custom Phoneme Override**: Use Markdown-like syntax to specify exact pronunciations: `[word](/phonemes/)`
- **Stress Pattern Control**: Automatic stress assignment with manual override capabilities
- **Apple Ecosystem Integration**: Uses Apple's Natural Language framework instead of external dependencies like SpaCy

## Architecture

MisakiSwift consists of several key components:

- **`EnglishG2P`**: Main conversion pipeline that orchestrates tokenization, lexicon lookup, and neural network fallback
- **`Lexicon`**: Dictionary-based pronunciation lookup using gold and silver dictionaries
- **`EnglishFallbackNetwork`**: Transformer-based BART model (a pure-Accelerate CPU port) for phoneme prediction for out-of-vocabulary words

## Key Differences from Python Misaki

1. **POS Tagging**: Uses Apple's `NaturalLanguage` framework instead of SpaCy for part-of-speech tagging
2. **Neural Network**: The BART-based fallback network is a pure-CPU Accelerate port (no GPU/MLX runtime)
3. **Resource Management**: Model weights and dictionaries are **not** bundled — the host injects them (see below)

## Dependencies

- **NaturalLanguage**: Apple's built-in framework for text processing and POS tagging
- **Accelerate**: Apple's built-in framework backing the BART fallback's matrix math

The package has no external package dependencies.

## Model Resources

`EnglishG2P` needs four data resources **per accent**:

- **`model.safetensors`**: BART fallback weights for phoneme prediction
- **`config.json`**: BART configuration (grapheme/phoneme vocabularies, layer sizes)
- **`gold.json`**: High-confidence pronunciation mappings
- **`silver.json`**: Additional pronunciation mappings with slightly lower confidence

These are **not bundled** with the package (~18 MB total). Instead, the host supplies them through a
`MisakiResourceLoading` injected into `EnglishG2P.init(british:unk:resources:)` — typically a
`MisakiDirectoryResources` pointing at an **accent-specific** directory the host downloaded (e.g. from the
Hugging Face repos `jonschneider/graphemes_to_phonemes_en_us` and `…_en_gb`). The library itself does no
downloading, so the host controls when assets are fetched. The US and GB assets are byte-identical to
those repos; the lexicons (`gold.json`, `silver.json`) are uploaded alongside them.

> **Breaking change:** `EnglishG2P.init` now requires a `resources:` argument and is `throws`. Previously
> the weights and dictionaries were bundled and loaded implicitly.
