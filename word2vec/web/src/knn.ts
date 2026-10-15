import type { Model } from './model'

export interface Neighbour {
  id: number
  word: string
  score: number
}

/** Top-k rows by cosine similarity to query, skipping special tokens, zero rows and `exclude`. */
export function knn(model: Model, query: Float32Array, k: number, exclude: ReadonlySet<number> = new Set()): Neighbour[] {
  const { q, dim, scales, invNorm, special, words } = model
  let qq = 0
  for (let j = 0; j < dim; j++) qq += query[j] * query[j]
  if (qq === 0 || k <= 0) return []
  const qInv = 1 / Math.sqrt(qq)
  const ids: number[] = []
  const scores: number[] = [] // descending, at most k long
  for (let i = 0, o = 0; i < words.length; i++, o += dim) {
    if (special[i] || invNorm[i] === 0 || exclude.has(i)) continue
    let d = 0
    for (let j = 0; j < dim; j++) d += q[o + j] * query[j]
    const s = d * scales[i] * invNorm[i] * qInv
    if (ids.length === k && s <= scores[k - 1]) continue
    let p = Math.min(ids.length, k - 1)
    while (p > 0 && scores[p - 1] < s) {
      scores[p] = scores[p - 1]
      ids[p] = ids[p - 1]
      p--
    }
    scores[p] = s
    ids[p] = i
  }
  return ids.map((id, r) => ({ id, word: words[id], score: scores[r] }))
}
