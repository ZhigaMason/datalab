import { describe, expect, test } from 'vitest'
import { evaluate, ExprError, isZero, parse, splitWords } from '../src/expr'
import { makeModel } from './helpers'

describe('parse', () => {
  test('signs and default coefficients', () => {
    expect(parse('king - man + woman')).toEqual([
      { word: 'king', coef: 1 },
      { word: 'man', coef: -1 },
      { word: 'woman', coef: 1 },
    ])
  })

  test('weights, case and spacing', () => {
    expect(parse('  0.5*Paris+berlin ')).toEqual([
      { word: 'paris', coef: 0.5 },
      { word: 'berlin', coef: 1 },
    ])
    expect(parse('-2 king')).toEqual([{ word: 'king', coef: -2 }])
    expect(parse('.5 * a - 1.5*b')).toEqual([
      { word: 'a', coef: 0.5 },
      { word: 'b', coef: -1.5 },
    ])
  })

  test('accepts the Unicode minus and dashes the page itself shows', () => {
    expect(parse('king − man – woman')).toEqual([
      { word: 'king', coef: 1 },
      { word: 'man', coef: -1 },
      { word: 'woman', coef: -1 },
    ])
  })

  test('a hyphen inside a word is part of the word', () => {
    expect(parse('well-known - x')).toEqual([
      { word: 'well-known', coef: 1 },
      { word: 'x', coef: -1 },
    ])
  })

  test.each(['', '   ', 'king man', 'king +', '2*', 'a * b'])('rejects %j', (src) => {
    expect(() => parse(src)).toThrow(ExprError)
  })
})

describe('evaluate', () => {
  // unit(a) = (1, 0), unit(b) = (0, 1)
  const m = makeModel(['<pad>', 'a', 'b'], [[0, 0], [10, 0], [0, 20]])

  test('sums unit vectors with coefficients', () => {
    const r = evaluate(m, 'a - 2*B')
    expect(r.terms.map((t) => t.id)).toEqual([1, 2])
    expect(Array.from(r.vector)).toEqual([1, -2])
  })

  test('names unknown words', () => {
    expect(() => evaluate(m, 'a + zzz')).toThrow('unknown word “zzz”')
  })

  test('hints at spaces when an unknown word contains a hyphen', () => {
    expect(() => evaluate(m, 'a-b')).toThrow(/spaces around -/)
  })
})

describe('isZero', () => {
  const m = makeModel(['<pad>', 'a', 'b'], [[0, 0], [10, 0], [0, 20]])
  const terms = [{ word: 'a', coef: 1, id: 1 }, { word: 'a', coef: -1, id: 1 }]

  test('treats exact and rounding-level cancellation as zero', () => {
    expect(isZero(evaluate(m, 'a - a'))).toBe(true)
    expect(isZero({ terms, vector: Float32Array.of(3e-8, -1e-8) })).toBe(true)
  })

  test('keeps genuine results', () => {
    expect(isZero(evaluate(m, 'a - b'))).toBe(false)
    expect(isZero({ terms, vector: Float32Array.of(0.01, 0) })).toBe(false)
  })
})

describe('splitWords', () => {
  test('splits on commas and whitespace, lowercases and deduplicates', () => {
    expect(splitWords('King, queen  king\nprince,')).toEqual(['king', 'queen', 'prince'])
    expect(splitWords('  ')).toEqual([])
  })
})
