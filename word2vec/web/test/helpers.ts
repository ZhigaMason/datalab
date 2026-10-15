import { Model } from '../src/model'

/** A Model from integer rows with one shared scale (rows[i] are the int8 values). */
export function makeModel(words: string[], rows: number[][], scale = 0.1): Model {
  return new Model(words, rows[0].length, Int8Array.from(rows.flat()), new Float32Array(words.length).fill(scale))
}
