import type { Model } from './model'

export interface Neighbour {
  id: number
  word: string
  score: number
}

export type Metric = 'cos' | 'l2'

/**
 * Top-k rows for the query, skipping special tokens, zero rows and `exclude`:
 * by cosine similarity (score descending) or by L2 distance between stored vectors (score ascending).
 */
export function knn(
  model: Model,
  query: Float32Array,
  k: number,
  exclude: ReadonlySet<number> = new Set(),
  metric: Metric = 'cos',
): Neighbour[] {
  const { q, dim, scales, invNorm, sqNorm, special, words } = model
  let qq = 0
  for (let j = 0; j < dim; j++) qq += query[j] * query[j]
  if ((metric === 'cos' && qq === 0) || k <= 0) return []
  const qInv = 1 / Math.sqrt(qq)
  const ids: number[] = []
  const keys: number[] = [] // descending (higher is better), at most k long
  for (let i = 0, o = 0; i < words.length; i++, o += dim) {
    if (special[i] || invNorm[i] === 0 || exclude.has(i)) continue
    let d = 0
    for (let j = 0; j < dim; j++) d += q[o + j] * query[j]
    // cosine, or minus the squared distance ‖r‖² − 2 r·q + ‖q‖²
    const key = metric === 'cos' ? d * scales[i] * invNorm[i] * qInv : -(sqNorm[i] - 2 * d * scales[i] + qq)
    if (ids.length === k && key <= keys[k - 1]) continue
    let p = Math.min(ids.length, k - 1)
    while (p > 0 && keys[p - 1] < key) {
      keys[p] = keys[p - 1]
      ids[p] = ids[p - 1]
      p--
    }
    keys[p] = key
    ids[p] = i
  }
  return ids.map((id, r) => ({ id, word: words[id], score: metric === 'cos' ? keys[r] : Math.sqrt(Math.max(0, -keys[r])) }))
}
