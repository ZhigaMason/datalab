import { readFileSync } from 'node:fs'
import { describe, expect, test } from 'vitest'
import { f32FromRaw, readInitializers } from '../src/onnx'

const file = readFileSync(new URL('./fixtures/tiny.onnx', import.meta.url))
const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)
const expected = JSON.parse(readFileSync(new URL('./fixtures/tiny.json', import.meta.url), 'utf8'))

describe('readInitializers', () => {
  test('finds the int8 table and the row scales', () => {
    const t = readInitializers(buf)
    const e = t.get('embeddings_int8')!, s = t.get('row_scales')!
    expect(e.dims).toEqual([5, 3])
    expect(e.dataType).toBe(3) // INT8
    expect(Array.from(new Int8Array(e.raw.buffer, e.raw.byteOffset, e.raw.byteLength))).toEqual(expected.int8)
    expect(s.dims).toEqual([5, 1])
    expect(s.dataType).toBe(1) // FLOAT
    expect(Array.from(f32FromRaw(s.raw))).toEqual(expected.scales.map(Math.fround))
  })

  test('throws on a truncated file', () => {
    expect(() => readInitializers(buf.slice(0, buf.byteLength >> 1))).toThrow(/truncated/)
  })
})

describe('f32FromRaw', () => {
  test('reads little-endian floats at an unaligned offset', () => {
    const bytes = new Uint8Array(9)
    const v = new DataView(bytes.buffer)
    v.setFloat32(1, 1.5, true)
    v.setFloat32(5, -2, true)
    expect(Array.from(f32FromRaw(bytes.subarray(1)))).toEqual([1.5, -2])
  })

  test('rejects a length that is not a multiple of 4', () => {
    expect(() => f32FromRaw(new Uint8Array(6))).toThrow(/multiple of 4/)
  })
})
