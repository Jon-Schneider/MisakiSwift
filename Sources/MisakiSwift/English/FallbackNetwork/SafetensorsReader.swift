import Foundation

/// Minimal reader for the safetensors format, supporting only what the fallback BART checkpoints use:
/// little-endian `F32` tensors. Layout: an 8-byte little-endian header length, a JSON header mapping
/// tensor names to `{"dtype", "shape", "data_offsets"}` (offsets relative to the end of the header),
/// then the raw tensor bytes.
enum SafetensorsReader {
  struct Tensor {
    let shape: [Int]
    let values: [Float]
  }

  enum ReadError: Error {
    case malformedHeader
    case unsupportedDType(String)
    case malformedTensorData(name: String)
  }

  static func read(contentsOf url: URL) throws -> [String: Tensor] {
    try read(Data(contentsOf: url))
  }

  static func read(_ data: Data) throws -> [String: Tensor] {
    guard data.count >= 8 else { throw ReadError.malformedHeader }

    let headerLength = data.prefix(8).withUnsafeBytes { buffer in
      UInt64(littleEndian: buffer.loadUnaligned(as: UInt64.self))
    }
    let dataStart = 8 + Int(headerLength)
    guard dataStart <= data.count,
          let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< dataStart)) as? [String: Any]
    else {
      throw ReadError.malformedHeader
    }

    var tensors = [String: Tensor]()
    for (name, value) in header {
      if name == "__metadata__" { continue }
      guard let entry = value as? [String: Any],
            let dtype = entry["dtype"] as? String,
            let shape = entry["shape"] as? [Int],
            let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
      else {
        throw ReadError.malformedTensorData(name: name)
      }
      guard dtype == "F32" else { throw ReadError.unsupportedDType(dtype) }

      let elementCount = shape.reduce(1, *)
      let start = dataStart + offsets[0]
      let end = dataStart + offsets[1]
      guard end <= data.count, end - start == elementCount * MemoryLayout<Float>.size else {
        throw ReadError.malformedTensorData(name: name)
      }

      var values = [Float](repeating: 0, count: elementCount)
      _ = values.withUnsafeMutableBytes { destination in
        data.copyBytes(to: destination, from: start ..< end)
      }
      tensors[name] = Tensor(shape: shape, values: values)
    }
    return tensors
  }
}
