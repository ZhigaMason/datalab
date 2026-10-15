import { readFileSync } from 'node:fs'
import { afterEach, describe, expect, test, vi } from 'vitest'
import { estimateMB, fetchBytes, LARGE_MB, Model, runName } from '../src/model'

const file = readFileSync(new URL('./fixtures/tiny.onnx', import.meta.url))
const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)
const words = ['<pad>', 'a', 'b', 'c', 'd']

describe('runName / estimateMB', () => {
  test('builds Hub directory names', () => {
    expect(runName({ lang: 'en', jepa: false, causal: false, dim: 300, window: 5 })).toBe('en-bidir-d300-w5-s0')
    expect(runName({ lang: 'cs', jepa: true, causal: true, dim: 64, window: 20 })).toBe('cs-causal-jepa-d64-w20-s0')
  })

  test('estimates sizes and puts the 20 MB line between d64 and d256', () => {
    expect(estimateMB(16)).toBeCloseTo(4.0, 1)
    expect(estimateMB(300)).toBeCloseTo(60.8, 1)
    expect(estimateMB(64)).toBeLessThan(LARGE_MB)
    expect(estimateMB(256)).toBeGreaterThan(LARGE_MB)
  })
})

describe('Model.fromOnnx', () => {
  const m = Model.fromOnnx(words, buf)

  test('exposes the table, lowercased lookup and special tokens', () => {
    expect(m.dim).toBe(3)
    expect(m.id('A')).toBe(1)
    expect(m.id('zzz')).toBeUndefined()
    expect(m.special[0]).toBe(1)
    expect(m.special[1]).toBe(0)
    expect(m.invNorm[0]).toBe(0) // the all-zero row
  })

  test('unit() returns unit vectors in the stored direction', () => {
    const u = m.unit(2) // weight row [1, 0.5, -0.25]
    expect(Math.hypot(...u)).toBeCloseTo(1, 6)
    const n = Math.hypot(1, 0.5, -0.25)
    expect(u[0]).toBeCloseTo(1 / n, 2)
    expect(u[2]).toBeCloseTo(-0.25 / n, 2)
  })

  test('rejects a vocab of the wrong length', () => {
    expect(() => Model.fromOnnx(words.slice(1), buf)).toThrow(/rows/)
  })
})

describe('fetchBytes', () => {
  afterEach(() => vi.unstubAllGlobals())

  test('collects the body and reports progress', async () => {
    vi.stubGlobal('fetch', async () => new Response(new Uint8Array([1, 2, 3])))
    const seen: number[] = []
    const out = await fetchBytes('https://x/y', 'y', (_, loaded) => seen.push(loaded))
    expect(Array.from(new Uint8Array(out))).toEqual([1, 2, 3])
    expect(seen.at(-1)).toBe(3)
  })

  test('names the URL on an HTTP error', async () => {
    vi.stubGlobal('fetch', async () => new Response('nope', { status: 404 }))
    await expect(fetchBytes('https://x/missing', 'm', () => {})).rejects.toThrow('HTTP 404 for https://x/missing')
  })
})
