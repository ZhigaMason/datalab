import { describe, expect, test } from 'vitest'
import { axisVector, evaluate, ExprError, isZero, parse, splitWords, ZERO_ID } from '../src/expr'
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

describe('axisVector', () => {
  // unit(a) = (1, 0), unit(b) = (0, 1)
  const m = makeModel(['<pad>', 'a', 'b'], [[0, 0], [10, 0], [0, 20]])

  test('points from the negative end to the positive end', () => {
    expect(Array.from(axisVector(m, 'a', 'b'))).toEqual(Array.from(evaluate(m, 'b - a').vector))
  })

  test('ends may be expressions', () => {
    expect(Array.from(axisVector(m, 'a', '2*b + a'))).toEqual([0, 2])
  })

  test('an empty end leaves the other end alone', () => {
    expect(Array.from(axisVector(m, '  ', 'b'))).toEqual([0, 1])
    expect(Array.from(axisVector(m, 'a', ''))).toEqual([-1, 0])
  })

  test('rejects two empty ends and ends that cancel', () => {
    expect(() => axisVector(m, '', ' ')).toThrow(ExprError)
    expect(() => axisVector(m, 'a', 'a')).toThrow(/zero vector/)
  })

  test('names a bad end', () => {
    expect(() => axisVector(m, 'zzz', 'a')).toThrow('unknown word “zzz”')
  })
})

describe('the zero vector 0', () => {
  const m = makeModel(['<pad>', 'a', 'b'], [[0, 0], [10, 0], [0, 20]])

  test('parses as a word, alone, in sums and with a weight', () => {
    expect(parse('0')).toEqual([{ word: '0', coef: 1 }])
    expect(parse('king - man + 0')).toEqual([
      { word: 'king', coef: 1 },
      { word: 'man', coef: -1 },
      { word: '0', coef: 1 },
    ])
    expect(parse('2*0')).toEqual([{ word: '0', coef: 2 }])
  })

  test('leaves weights and other numbers alone', () => {
    expect(parse('0.5*king')).toEqual([{ word: 'king', coef: 0.5 }])
    expect(parse('0 king')).toEqual([{ word: 'king', coef: 0 }])
    expect(() => parse('10')).toThrow(ExprError)
    expect(() => parse('0.0')).toThrow(ExprError)
  })

  test('evaluates to nothing added', () => {
    const z = evaluate(m, '0')
    expect(z.terms).toEqual([{ word: '0', coef: 1, id: ZERO_ID }])
    expect(Array.from(z.vector)).toEqual([0, 0])
    expect(isZero(z)).toBe(true)
    expect(Array.from(evaluate(m, 'a + 0').vector)).toEqual([1, 0])
  })

  test('works as an axis end', () => {
    expect(Array.from(axisVector(m, '0', 'b'))).toEqual([0, 1])
    expect(() => axisVector(m, '0', '0')).toThrow(/zero vector/)
  })
})

describe('splitWords', () => {
  test('splits on commas and whitespace, lowercases and deduplicates', () => {
    expect(splitWords('King, queen  king\nprince,')).toEqual(['king', 'queen', 'prince'])
    expect(splitWords('  ')).toEqual([])
  })
})
