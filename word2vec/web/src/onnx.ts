/** Just enough protobuf to read tensor initializers out of an ONNX ModelProto. */

export interface Tensor {
  name: string
  dims: number[]
  dataType: number
  raw: Uint8Array
}

class Reader {
  pos = 0
  constructor(readonly buf: Uint8Array) {}

  get done(): boolean {
    return this.pos >= this.buf.length
  }

  varint(): number {
    let result = 0
    let mul = 1
    for (;;) {
      if (this.pos >= this.buf.length) throw new Error('truncated ONNX file (varint)')
      const b = this.buf[this.pos++]
      result += (b & 0x7f) * mul
      if (b < 0x80) return result
      mul *= 128
    }
  }

  bytes(): Uint8Array {
    const len = this.varint()
    const start = this.pos
    this.advance(len)
    return this.buf.subarray(start, start + len)
  }

  advance(n: number): void {
    if (this.pos + n > this.buf.length) throw new Error('truncated ONNX file')
    this.pos += n
  }

  skip(wire: number): void {
    if (wire === 0) this.varint()
    else if (wire === 1) this.advance(8)
    else if (wire === 2) this.bytes()
    else if (wire === 5) this.advance(4)
    else throw new Error(`unsupported protobuf wire type ${wire}`)
  }
}

/** Calls onField for every field of a message; fields it does not consume (returns false) are skipped. */
function readFields(buf: Uint8Array, onField: (field: number, wire: number, r: Reader) => boolean): void {
  const r = new Reader(buf)
  while (!r.done) {
    const key = r.varint()
    const wire = key % 8
    if (!onField(Math.floor(key / 8), wire, r)) r.skip(wire)
  }
}

function readTensor(buf: Uint8Array): Tensor {
  const t: Tensor = { name: '', dims: [], dataType: 0, raw: new Uint8Array(0) }
  readFields(buf, (field, wire, r) => {
    if (field === 1 && wire === 0) t.dims.push(r.varint())
    else if (field === 1 && wire === 2) {
      const packed = new Reader(r.bytes())
      while (!packed.done) t.dims.push(packed.varint())
    } else if (field === 2 && wire === 0) t.dataType = r.varint()
    else if (field === 8 && wire === 2) t.name = new TextDecoder().decode(r.bytes())
    else if (field === 9 && wire === 2) t.raw = r.bytes()
    else return false
    return true
  })
  return t
}

/** ModelProto.graph (7) → GraphProto.initializer (5) → TensorProto, keyed by name. */
export function readInitializers(buf: ArrayBuffer): Map<string, Tensor> {
  const out = new Map<string, Tensor>()
  readFields(new Uint8Array(buf), (field, wire, r) => {
    if (field !== 7 || wire !== 2) return false
    readFields(r.bytes(), (gField, gWire, g) => {
      if (gField !== 5 || gWire !== 2) return false
      const t = readTensor(g.bytes())
      out.set(t.name, t)
      return true
    })
    return true
  })
  return out
}

/** Little-endian float32 values from bytes at any offset (raw_data is not 4-byte aligned in general). */
export function f32FromRaw(raw: Uint8Array): Float32Array {
  if (raw.byteLength % 4) throw new Error('float32 data length is not a multiple of 4')
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  const out = new Float32Array(raw.byteLength / 4)
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true)
  return out
}
