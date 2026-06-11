import Accelerate
import Foundation

/// CPU implementation of the tiny BART encoder-decoder used to phonemize out-of-vocabulary words.
///
/// This replaces the previous MLX implementation with plain Accelerate so the package carries no GPU
/// runtime: iOS kills processes that submit Metal work in the background, and narration synthesizes
/// while backgrounded. The checkpoints are small enough (d_model 128, one encoder and one decoder
/// layer, vocab 63, ~3 MB fp32) that a word decodes in well under a millisecond on the CPU.
///
/// The forward pass mirrors the original MLX module graph *exactly* — including its quirks — so the
/// generated phonemes are unchanged (verified by committed fixtures captured from the MLX
/// implementation running on the CPU):
/// - positions are offset by +2 into the learned positional embeddings (BART convention),
/// - layer norm is post-norm with the MLXNN default epsilon of 1e-5,
/// - gelu is the exact erf form, not the tanh approximation,
/// - the decoder runs with **no causal self-attention mask** and re-decodes the full prefix each step
///   (only the last position's logits are read, so greedy decoding still works),
/// - decoding is greedy argmax (first index on ties) up to 50 steps with a forced EOS on the last.
nonisolated final class BARTFallbackModel {

  // MARK: Lifecycle

  init(config: BARTConfig, weights: [String: SafetensorsReader.Tensor]) {
    self.config = config

    sharedEmbedding = Self.matrix(weights, "model.shared.weight")
    encoderPositionalEmbedding = Self.matrix(weights, "model.encoder.embed_positions.weight")
    decoderPositionalEmbedding = Self.matrix(weights, "model.decoder.embed_positions.weight")

    encoderEmbeddingNorm = LayerNorm(weights, "model.encoder.layernorm_embedding")
    decoderEmbeddingNorm = LayerNorm(weights, "model.decoder.layernorm_embedding")

    encoderLayers = (0 ..< config.encoderLayers).map { index in
      EncoderLayer(weights: weights, modelKey: "model.encoder.layers.\(index)", heads: config.encoderAttentionHeads)
    }
    decoderLayers = (0 ..< config.decoderLayers).map { index in
      DecoderLayer(weights: weights, modelKey: "model.decoder.layers.\(index)", heads: config.decoderAttentionHeads)
    }

    finalLogitsBias = weights["final_logits_bias"]?.values ?? [Float](repeating: 0, count: config.vocabSize)
  }

  // MARK: Internal

  /// Greedy-decodes phoneme token ids for the given grapheme token ids. Mirrors the MLX
  /// implementation's `generate(inputIds:maxLength:temperature:)` with its default arguments.
  func generate(inputIds: [Int], maxLength: Int = 50) -> [Int] {
    let encoderOutput = encode(inputIds)

    var decoderInput = [config.bosTokenId]
    var generatedTokens = [Int]()

    for i in 0 ..< maxLength {
      if i == maxLength - 1 {
        generatedTokens.append(config.eosTokenId)
        break
      }

      let logits = decode(decoderInput, encoderOutput: encoderOutput)
      let lastRow = Array(logits.values[(logits.rows - 1) * logits.cols ..< logits.rows * logits.cols])
      let nextToken = Self.argMax(lastRow)

      if nextToken == config.eosTokenId {
        break
      }

      generatedTokens.append(nextToken)
      decoderInput.append(nextToken)
    }

    return generatedTokens
  }

  // MARK: Private

  /// Row-major 2-D float buffer.
  private struct Matrix {
    var rows: Int
    var cols: Int
    var values: [Float]
  }

  private struct LayerNorm {
    let weight: [Float]
    let bias: [Float]

    init(_ weights: [String: SafetensorsReader.Tensor], _ key: String) {
      weight = weights["\(key).weight"]?.values ?? []
      bias = weights["\(key).bias"]?.values ?? []
    }

    /// Post-norm with biased variance and the MLXNN default epsilon.
    func callAsFunction(_ x: Matrix) -> Matrix {
      let eps: Float = 1e-5
      var output = x
      for row in 0 ..< x.rows {
        let range = row * x.cols ..< (row + 1) * x.cols
        var mean: Float = 0
        vDSP_meanv(Array(x.values[range]), 1, &mean, vDSP_Length(x.cols))
        var variance: Float = 0
        for i in range {
          let centered = x.values[i] - mean
          variance += centered * centered
        }
        variance /= Float(x.cols)
        let inverseDeviation = 1 / (variance + eps).squareRoot()
        for (column, i) in range.enumerated() {
          output.values[i] = (x.values[i] - mean) * inverseDeviation * weight[column] + bias[column]
        }
      }
      return output
    }
  }

  private struct Linear {
    /// `[outputDimension, inputDimension]`, the Hugging Face layout.
    let weight: Matrix
    let bias: [Float]?

    init(_ weights: [String: SafetensorsReader.Tensor], _ key: String) {
      weight = BARTFallbackModel.matrix(weights, "\(key).weight")
      bias = weights["\(key).bias"]?.values
    }

    /// `x @ weightᵀ + bias`
    func callAsFunction(_ x: Matrix) -> Matrix {
      var output = Matrix(rows: x.rows, cols: weight.rows, values: [Float](repeating: 0, count: x.rows * weight.rows))
      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        Int32(x.rows), Int32(weight.rows), Int32(x.cols),
        1, x.values, Int32(x.cols),
        weight.values, Int32(weight.cols),
        0, &output.values, Int32(weight.rows)
      )
      if let bias {
        for row in 0 ..< output.rows {
          for column in 0 ..< output.cols {
            output.values[row * output.cols + column] += bias[column]
          }
        }
      }
      return output
    }
  }

  private struct Attention {
    let heads: Int
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let outProj: Linear

    init(weights: [String: SafetensorsReader.Tensor], modelKey: String, heads: Int) {
      self.heads = heads
      qProj = Linear(weights, "\(modelKey).q_proj")
      kProj = Linear(weights, "\(modelKey).k_proj")
      vProj = Linear(weights, "\(modelKey).v_proj")
      outProj = Linear(weights, "\(modelKey).out_proj")
    }

    /// Scaled dot-product attention without any mask, matching the MLX implementation (which never
    /// passes one — see the class comment on the missing causal mask).
    func callAsFunction(query: Matrix, keysAndValues: Matrix) -> Matrix {
      let q = qProj(query)
      let k = kProj(keysAndValues)
      let v = vProj(keysAndValues)

      let headDim = q.cols / heads
      let scale = Float(1.0 / Double(headDim).squareRoot())
      var output = Matrix(rows: q.rows, cols: q.cols, values: [Float](repeating: 0, count: q.rows * q.cols))

      for head in 0 ..< heads {
        let columns = head * headDim ..< (head + 1) * headDim
        let qHead = BARTFallbackModel.columns(q, columns)
        let kHead = BARTFallbackModel.columns(k, columns)
        let vHead = BARTFallbackModel.columns(v, columns)

        // scores[q.rows, k.rows] = qHead @ kHeadᵀ * scale
        var scores = Matrix(rows: qHead.rows, cols: kHead.rows, values: [Float](repeating: 0, count: qHead.rows * kHead.rows))
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasTrans,
          Int32(qHead.rows), Int32(kHead.rows), Int32(headDim),
          scale, qHead.values, Int32(headDim),
          kHead.values, Int32(headDim),
          0, &scores.values, Int32(kHead.rows)
        )
        BARTFallbackModel.softmaxRows(&scores)

        var headOutput = Matrix(rows: qHead.rows, cols: headDim, values: [Float](repeating: 0, count: qHead.rows * headDim))
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans,
          Int32(scores.rows), Int32(headDim), Int32(scores.cols),
          1, scores.values, Int32(scores.cols),
          vHead.values, Int32(headDim),
          0, &headOutput.values, Int32(headDim)
        )
        for row in 0 ..< output.rows {
          for (offset, column) in columns.enumerated() {
            output.values[row * output.cols + column] = headOutput.values[row * headDim + offset]
          }
        }
      }

      return outProj(output)
    }
  }

  private struct FeedForward {
    let linear1: Linear
    let linear2: Linear

    init(weights: [String: SafetensorsReader.Tensor], modelKey: String) {
      linear1 = Linear(weights, "\(modelKey).fc1")
      linear2 = Linear(weights, "\(modelKey).fc2")
    }

    func callAsFunction(_ x: Matrix) -> Matrix {
      var hidden = linear1(x)
      BARTFallbackModel.gelu(&hidden)
      return linear2(hidden)
    }
  }

  private struct EncoderLayer {
    let selfAttn: Attention
    let selfAttnNorm: LayerNorm
    let ffn: FeedForward
    let ffnNorm: LayerNorm

    init(weights: [String: SafetensorsReader.Tensor], modelKey: String, heads: Int) {
      selfAttn = Attention(weights: weights, modelKey: "\(modelKey).self_attn", heads: heads)
      selfAttnNorm = LayerNorm(weights, "\(modelKey).self_attn_layer_norm")
      ffn = FeedForward(weights: weights, modelKey: modelKey)
      ffnNorm = LayerNorm(weights, "\(modelKey).final_layer_norm")
    }

    func callAsFunction(_ x: Matrix) -> Matrix {
      let attnOutput = selfAttn(query: x, keysAndValues: x)
      var output = selfAttnNorm(BARTFallbackModel.sum(x, attnOutput))
      let ffnOutput = ffn(output)
      output = ffnNorm(BARTFallbackModel.sum(output, ffnOutput))
      return output
    }
  }

  private struct DecoderLayer {
    let selfAttn: Attention
    let selfAttnNorm: LayerNorm
    let crossAttn: Attention
    let crossAttnNorm: LayerNorm
    let ffn: FeedForward
    let ffnNorm: LayerNorm

    init(weights: [String: SafetensorsReader.Tensor], modelKey: String, heads: Int) {
      selfAttn = Attention(weights: weights, modelKey: "\(modelKey).self_attn", heads: heads)
      selfAttnNorm = LayerNorm(weights, "\(modelKey).self_attn_layer_norm")
      crossAttn = Attention(weights: weights, modelKey: "\(modelKey).encoder_attn", heads: heads)
      crossAttnNorm = LayerNorm(weights, "\(modelKey).encoder_attn_layer_norm")
      ffn = FeedForward(weights: weights, modelKey: modelKey)
      ffnNorm = LayerNorm(weights, "\(modelKey).final_layer_norm")
    }

    func callAsFunction(_ x: Matrix, encoderOutput: Matrix) -> Matrix {
      let attnOutput = selfAttn(query: x, keysAndValues: x)
      var output = selfAttnNorm(BARTFallbackModel.sum(x, attnOutput))
      let crossOutput = crossAttn(query: output, keysAndValues: encoderOutput)
      output = crossAttnNorm(BARTFallbackModel.sum(output, crossOutput))
      let ffnOutput = ffn(output)
      output = ffnNorm(BARTFallbackModel.sum(output, ffnOutput))
      return output
    }
  }

  private let config: BARTConfig
  private let sharedEmbedding: Matrix
  private let encoderPositionalEmbedding: Matrix
  private let decoderPositionalEmbedding: Matrix
  private let encoderEmbeddingNorm: LayerNorm
  private let decoderEmbeddingNorm: LayerNorm
  private let encoderLayers: [EncoderLayer]
  private let decoderLayers: [DecoderLayer]
  private let finalLogitsBias: [Float]

  private static func matrix(_ weights: [String: SafetensorsReader.Tensor], _ key: String) -> Matrix {
    guard let tensor = weights[key], tensor.shape.count == 2 else {
      return Matrix(rows: 0, cols: 0, values: [])
    }
    return Matrix(rows: tensor.shape[0], cols: tensor.shape[1], values: tensor.values)
  }

  private static func columns(_ x: Matrix, _ range: Range<Int>) -> Matrix {
    var output = Matrix(rows: x.rows, cols: range.count, values: [Float](repeating: 0, count: x.rows * range.count))
    for row in 0 ..< x.rows {
      for (offset, column) in range.enumerated() {
        output.values[row * range.count + offset] = x.values[row * x.cols + column]
      }
    }
    return output
  }

  private static func sum(_ a: Matrix, _ b: Matrix) -> Matrix {
    var output = a
    vDSP_vadd(a.values, 1, b.values, 1, &output.values, 1, vDSP_Length(a.values.count))
    return output
  }

  /// Exact erf-form gelu, `x · ½(1 + erf(x/√2))`, matching MLXNN's `gelu` (not the tanh approximation).
  private static func gelu(_ x: inout Matrix) {
    let inverseSqrtTwo = Float(1.0 / 2.0.squareRoot())
    for i in 0 ..< x.values.count {
      x.values[i] = x.values[i] * 0.5 * (1 + erff(x.values[i] * inverseSqrtTwo))
    }
  }

  private static func softmaxRows(_ x: inout Matrix) {
    for row in 0 ..< x.rows {
      let range = row * x.cols ..< (row + 1) * x.cols
      var maximum: Float = -.infinity
      for i in range {
        maximum = max(maximum, x.values[i])
      }
      var total: Float = 0
      for i in range {
        let value = expf(x.values[i] - maximum)
        x.values[i] = value
        total += value
      }
      for i in range {
        x.values[i] /= total
      }
    }
  }

  private static func argMax(_ values: [Float]) -> Int {
    var best = 0
    for i in 1 ..< values.count where values[i] > values[best] {
      best = i
    }
    return best
  }

  /// Token embeddings plus learned positional embeddings (offset by +2, the BART convention), then
  /// the embedding layer norm.
  private func embed(_ tokenIds: [Int], positional: Matrix, norm: LayerNorm) -> Matrix {
    let d = sharedEmbedding.cols
    var hidden = Matrix(rows: tokenIds.count, cols: d, values: [Float](repeating: 0, count: tokenIds.count * d))
    for (row, tokenId) in tokenIds.enumerated() {
      let position = row + 2
      for column in 0 ..< d {
        hidden.values[row * d + column] =
          sharedEmbedding.values[tokenId * d + column] + positional.values[position * d + column]
      }
    }
    return norm(hidden)
  }

  private func encode(_ inputIds: [Int]) -> Matrix {
    var hidden = embed(inputIds, positional: encoderPositionalEmbedding, norm: encoderEmbeddingNorm)
    for layer in encoderLayers {
      hidden = layer(hidden)
    }
    return hidden
  }

  /// Returns logits `[inputIds.count, vocab]`: the decoder stack followed by the tied-embedding LM
  /// head plus `final_logits_bias` (zeros in these checkpoints, applied anyway to match the original).
  private func decode(_ inputIds: [Int], encoderOutput: Matrix) -> Matrix {
    var hidden = embed(inputIds, positional: decoderPositionalEmbedding, norm: decoderEmbeddingNorm)
    for layer in decoderLayers {
      hidden = layer(hidden, encoderOutput: encoderOutput)
    }

    var logits = Matrix(rows: hidden.rows, cols: sharedEmbedding.rows, values: [Float](repeating: 0, count: hidden.rows * sharedEmbedding.rows))
    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasTrans,
      Int32(hidden.rows), Int32(sharedEmbedding.rows), Int32(hidden.cols),
      1, hidden.values, Int32(hidden.cols),
      sharedEmbedding.values, Int32(sharedEmbedding.cols),
      0, &logits.values, Int32(sharedEmbedding.rows)
    )
    for row in 0 ..< logits.rows {
      for column in 0 ..< logits.cols {
        logits.values[row * logits.cols + column] += finalLogitsBias[column]
      }
    }
    return logits
  }
}
