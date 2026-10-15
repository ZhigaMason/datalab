import { expect, test } from 'vitest'
import { evaluate } from '../src/expr'
import { knn } from '../src/knn'
import { loadModel } from '../src/model'

test.runIf(process.env.LIVE)('king - man + woman ≈ queen on the published en d300 model', async () => {
  const m = await loadModel({ lang: 'en', jepa: false, causal: false, dim: 300, window: 5 }, () => {})
  expect(m.words.length).toBe(200_000)
  const r = evaluate(m, 'king - man + woman')
  const t0 = performance.now()
  const top = knn(m, r.vector, 10, new Set(r.terms.map((t) => t.id)))
  console.log(`knn over 200k × 300: ${(performance.now() - t0).toFixed(0)} ms`, top.map((n) => n.word))
  expect(top.map((n) => n.word)).toContain('queen')
}, 180_000)
