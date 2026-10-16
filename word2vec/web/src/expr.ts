import type { Model } from './model'

export interface Term {
  word: string
  coef: number
}

export interface Resolved {
  terms: (Term & { id: number })[]
  vector: Float32Array
  /** Σ |coef| · ‖term vector‖ — the length the sum could have had; the yardstick for isZero. */
  scale: number
}

export class ExprError extends Error {}

/** The word `0` stands for the zero vector; its terms carry this id instead of a vocab row. */
export const ZERO_ID = -1

// [sign] [number [*]] word — a word cannot start with a digit, '.', '-', '+' or '*', except a standalone 0
const TERM = /\s*([+-])?\s*(?:(\d+(?:\.\d+)?|\.\d+)\s*\*?\s*)?([^\s+*\d.\-][^\s+*]*|(?<![\d.])0(?![\d.]))\s*/y

export function parse(src: string): Term[] {
  src = src.replace(/[−–—]/g, '-') // the page itself shows U+2212 in hints and labels
  if (!src.trim()) throw new ExprError('empty expression')
  const re = new RegExp(TERM)
  const terms: Term[] = []
  let pos = 0
  while (pos < src.length) {
    re.lastIndex = pos
    const m = re.exec(src)
    if (!m || (terms.length > 0 && !m[1])) throw new ExprError(`cannot parse “${src.slice(pos).trim()}”`)
    terms.push({ word: m[3].toLowerCase(), coef: (m[1] === '-' ? -1 : 1) * (m[2] ? Number(m[2]) : 1) })
    pos = re.lastIndex
  }
  return terms
}

/** Σ coef · unit(word), or Σ coef · stored vector(word) when `raw` (for L2 search). */
export function evaluate(model: Model, src: string, raw = false): Resolved {
  const terms = parse(src).map((t) => {
    if (t.word === '0') return { ...t, id: ZERO_ID }
    const id = model.id(t.word)
    if (id === undefined) {
      const hint = t.word.includes('-') ? ' (put spaces around - to subtract)' : ''
      throw new ExprError(`unknown word “${t.word}”${hint}`)
    }
    return { ...t, id }
  })
  const vector = new Float32Array(model.dim)
  let scale = 0
  for (const t of terms) {
    if (t.id === ZERO_ID) continue
    const u = raw ? model.vector(t.id) : model.unit(t.id)
    for (let j = 0; j < model.dim; j++) vector[j] += t.coef * u[j]
    scale += Math.abs(t.coef) * (raw ? Math.sqrt(model.sqNorm[t.id]) : 1)
  }
  return { terms, vector, scale }
}

/** True when the terms cancel out (up to float32 rounding), so the result has no meaningful direction. */
export function isZero(r: Resolved): boolean {
  let n = 0
  for (const x of r.vector) n += x * x
  return Math.sqrt(n) <= 1e-4 * r.scale
}

/** Axis pointing from the `neg` end to the `pos` end; either end (not both) may be empty. */
export function axisVector(model: Model, neg: string, pos: string): Float32Array {
  const vector = new Float32Array(model.dim)
  const terms: Resolved['terms'] = []
  let scale = 0
  ;[neg, pos].forEach((src, i) => {
    if (!src.trim()) return
    const r = evaluate(model, src)
    const sign = i === 0 ? -1 : 1
    for (let j = 0; j < model.dim; j++) vector[j] += sign * r.vector[j]
    terms.push(...r.terms)
    scale += r.scale
  })
  if (terms.length === 0) throw new ExprError('both ends are empty')
  if (isZero({ terms, vector, scale })) throw new ExprError('the two ends cancel out to the zero vector')
  return vector
}

export function splitWords(src: string): string[] {
  return [...new Set(src.toLowerCase().split(/[\s,]+/).filter(Boolean))]
}
