import { describe, expect, test } from 'vitest'
import { knn } from '../src/knn'
import { Model } from '../src/model'
import { makeModel } from './helpers'

// a = (1,0), b = (0,1), c = 45°, d = (-1,0); <pad> is special and zero
const m = makeModel(['<pad>', 'a', 'b', 'c', 'd'], [[0, 0], [10, 0], [0, 10], [7, 7], [-10, 0]])
const q = Float32Array.of(2, 0)

describe('knn', () => {
  test('ranks by cosine', () => {
    const r = knn(m, q, 2)
    expect(r.map((n) => n.word)).toEqual(['a', 'c'])
    expect(r[0].score).toBeCloseTo(1, 6)
    expect(r[1].score).toBeCloseTo(Math.SQRT1_2, 6)
  })

  test('honours exclusions and never returns special tokens', () => {
    expect(knn(m, q, 2, new Set([1])).map((n) => n.word)).toEqual(['c', 'b'])
    expect(knn(m, q, 10).map((n) => n.word)).toEqual(['a', 'c', 'b', 'd'])
  })

  test('returns nothing for a zero query', () => {
    expect(knn(m, new Float32Array(2), 5)).toEqual([])
  })

  test('matches a float brute force on random data', () => {
    let seed = 12345
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31) * 2 - 1
    const V = 60, dim = 8
    const words = Array.from({ length: V }, (_, i) => (i === 0 ? '<pad>' : `w${i}`))
    const big = new Model(words, dim, Int8Array.from({ length: V * dim }, () => Math.round(rnd() * 127)),
      Float32Array.from({ length: V }, () => 0.01 + Math.abs(rnd())))
    const query = Float32Array.from({ length: dim }, rnd)
    const qn = Math.hypot(...query)
    const ref = words.map((_, i) => ({ i, s: big.unit(i).reduce((acc, x, j) => acc + x * query[j], 0) / qn }))
      .slice(1).sort((a, b) => b.s - a.s).slice(0, 5)
    const got = knn(big, query, 5)
    expect(got.map((n) => n.id)).toEqual(ref.map((r) => r.i))
    got.forEach((n, r) => expect(n.score).toBeCloseTo(ref[r].s, 5))
  })
})
