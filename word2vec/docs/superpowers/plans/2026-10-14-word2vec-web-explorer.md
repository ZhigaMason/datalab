# Word2vec Web Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A static GitHub Pages playground that downloads an int8 word2vec table from Hugging Face on demand and offers vector arithmetic with k-NN plus a PCA map, and a 2D projection onto user-chosen axes.

**Architecture:** Vite + TypeScript with zero runtime dependencies. A minimal protobuf reader pulls the `embeddings_int8` and `row_scales` initializers out of `model_int8.onnx`; the table stays int8 in memory and k-NN scans it on the main thread. Pure modules (`onnx`, `model`, `expr`, `knn`, `geom`) are unit-tested with Vitest; `plot.ts` (hand-written SVG) and `main.ts` (DOM wiring) are checked by a build plus a manual run. A GitHub Actions workflow deploys `dist/` to Pages.

**Tech Stack:** TypeScript 7, Vite 8, Vitest 5 (dev dependencies only), Node 22, GitHub Actions Pages deploy. Python (existing `w2v.export`) only to generate a test fixture.

**Spec:** `word2vec/docs/superpowers/specs/2026-10-14-word2vec-web-explorer-design.md`

## Global Constraints

- Zero runtime dependencies: `package.json` has only `devDependencies` (`vite`, `vitest`, `typescript`).
- No UI framework, no chart library, no onnxruntime-web, no Web Worker, no autocomplete, no URL state, no Cache Storage.
- Model URLs: `https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk/resolve/main/` + `vocab/<lang>.json` and `<run>/model_int8.onnx`.
- Run name: `<lang>-<bidir|causal>[-jepa]-d<dim>-w<window>-s0`; lang ∈ {en, cs, uk}, dim ∈ {16, 64, 256, 300}, window ∈ {5, 10, 20}.
- Default model: `en-bidir-d300-w5`.
- Size estimate `V·(dim + 4)` bytes, V = 200 000. Warning sign ⚠ when estimate > 20 MB, tooltip `Large download (≈N MB), so consider a smaller dim on mobile data.` The warning is informational and never blocks loading.
- All vocabs are lowercase: every looked-up word is lowercased. Special tokens (`<…>`) are never returned as neighbours.
- Expression terms are L2-normalized before summing.
- Vite `base: '/datalab/'`; site URL `https://zhigamason.github.io/datalab/`.
- Plot colours (validated with the dataviz validator, light/dark): term/word `#2a78d6`/`#3987e5`, result `#eb6834`/`#d95926`, neighbour `#1baf7a`/`#199e70`; surface `#fcfcfb`/`#1a1a19`. Every point carries a direct text label (relief for the aqua contrast warning); text uses ink colours, never series colours.
- File paths are relative to the repo root `datalab/`; run `npm`/`npx` commands from `word2vec/web` and `uv` commands from `word2vec/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Float32 `raw_data` that starts at an odd byte offset inside the protobuf → scales read correctly (no `RangeError` from a misaligned `Float32Array`). Pinned in Task 1 (`f32FromRaw` at offset 1).
2. Hub returns 404 / network error → an error naming the URL, and the failed vocab promise is not cached. Pinned in Task 2 (`fetchBytes` 404 test).
3. Expressions that cancel (`king - king`) or that write subtraction without spaces (`woman-man`) → zero-vector message / a hint about spaces, never NaN neighbours. Pinned in Task 3 (hint test) and Task 4 (zero query returns `[]`).
4. y axis parallel to x with "make y ⊥ x" on (`x = man`, `y = 2*man`) → clear "parallel" error, not exploding coordinates. Pinned in Task 4.
5. Messy word lists (commas, newlines, capitals, duplicates) → cleaned, deduplicated, lowercased. Pinned in Task 3 (`splitWords`).

---

## File Structure

```
.github/workflows/pages.yml              # Task 6: build + deploy to Pages
word2vec/scripts/make_web_fixture.py     # Task 1: writes the tiny ONNX test fixture
word2vec/README.md                       # Task 6: "Web explorer" section
word2vec/web/
  .gitignore  package.json  package-lock.json  tsconfig.json  vite.config.ts   # Task 1
  index.html                             # Task 5
  src/onnx.ts                            # Task 1: protobuf → initializers
  src/model.ts                           # Task 2: Model class, run names, downloads
  src/expr.ts                            # Task 3: expression parser/evaluator, word lists
  src/knn.ts  src/geom.ts                # Task 4: cosine top-k, PCA, axis projection
  src/plot.ts  src/style.css  src/main.ts  # Task 5: SVG scatter, styles, wiring
  test/fixtures/tiny.onnx  tiny.json     # Task 1 (generated, committed)
  test/helpers.ts                        # Task 3
  test/onnx.test.ts  model.test.ts  expr.test.ts  knn.test.ts  geom.test.ts  live.test.ts
```

---

### Task 1: Scaffold the web project and the ONNX initializer reader

**Files:**
- Create: `word2vec/scripts/make_web_fixture.py`
- Create: `word2vec/web/.gitignore`, `word2vec/web/package.json`, `word2vec/web/tsconfig.json`, `word2vec/web/vite.config.ts`
- Create: `word2vec/web/src/onnx.ts`
- Create: `word2vec/web/test/onnx.test.ts`
- Generated: `word2vec/web/test/fixtures/tiny.onnx`, `word2vec/web/test/fixtures/tiny.json`

**Interfaces:**
- Consumes: `w2v.export.export_onnx(weight, path)` and `w2v.export.quantize_int8(weight) -> (q [V,d] int8, scale [V,1] float32)` (existing Python).
- Produces:
  - `interface Tensor { name: string; dims: number[]; dataType: number; raw: Uint8Array }`
  - `readInitializers(buf: ArrayBuffer): Map<string, Tensor>` — throws `Error` whose message contains `truncated` on a cut buffer.
  - `f32FromRaw(raw: Uint8Array): Float32Array` — little-endian, any byte offset.

- [ ] **Step 1: Write the fixture generator**

`word2vec/scripts/make_web_fixture.py`:

```python
"""Write web/test/fixtures/tiny.onnx and tiny.json (the expected int8 rows and scales) for the web tests."""
import json
from pathlib import Path

import torch

from w2v.export import export_onnx, quantize_int8

OUT = Path(__file__).resolve().parents[1] / "web" / "test" / "fixtures"


def main():
    weight = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [1.0, 0.5, -0.25], [-2.0, 1.0, 0.0], [0.3, 0.3, 0.3]])
    OUT.mkdir(parents=True, exist_ok=True)
    export_onnx(weight, OUT / "tiny.onnx")
    q, scale = quantize_int8(weight)
    (OUT / "tiny.json").write_text(json.dumps({"dims": list(q.shape), "int8": q.flatten().tolist(),
                                               "scales": scale.flatten().tolist()}))


if __name__ == "__main__":
    main()
```

Run from `word2vec/`: `uv run python -m scripts.make_web_fixture`
Expected: `web/test/fixtures/tiny.onnx` (a few hundred bytes) and `tiny.json` with `"dims": [5, 3]`.

- [ ] **Step 2: Scaffold the npm project**

`word2vec/web/.gitignore`:

```
node_modules/
dist/
```

`word2vec/web/package.json`:

```json
{
  "name": "word2vec-explorer",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  }
}
```

Then from `word2vec/web`: `npm install -D vite vitest typescript` (creates `package-lock.json` and the `devDependencies` block).

`word2vec/web/tsconfig.json` (tests are type-stripped by Vitest and deliberately left out of `tsc`, so no `@types/node` is needed):

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vite/client"],
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true
  },
  "include": ["src"]
}
```

`word2vec/web/vite.config.ts`:

```ts
import { defineConfig } from 'vite'

export default defineConfig({ base: '/datalab/' })
```

- [ ] **Step 3: Write the failing test**

`word2vec/web/test/onnx.test.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `npx vitest run test/onnx.test.ts`
Expected: FAIL, cannot resolve `../src/onnx`.

- [ ] **Step 5: Implement `src/onnx.ts`**

```ts
/** Just enough protobuf to read tensor initializers out of an ONNX ModelProto. */

export interface Tensor {
  name: string
  dims: number[]
  dataType: number
  raw: Uint8Array
}

class Reader {
  pos = 0
  constructor(readonly buf: Uint8Array) {}

  get done(): boolean {
    return this.pos >= this.buf.length
  }

  varint(): number {
    let result = 0
    let mul = 1
    for (;;) {
      if (this.pos >= this.buf.length) throw new Error('truncated ONNX file (varint)')
      const b = this.buf[this.pos++]
      result += (b & 0x7f) * mul
      if (b < 0x80) return result
      mul *= 128
    }
  }

  bytes(): Uint8Array {
    const len = this.varint()
    const start = this.pos
    this.advance(len)
    return this.buf.subarray(start, start + len)
  }

  advance(n: number): void {
    if (this.pos + n > this.buf.length) throw new Error('truncated ONNX file')
    this.pos += n
  }

  skip(wire: number): void {
    if (wire === 0) this.varint()
    else if (wire === 1) this.advance(8)
    else if (wire === 2) this.bytes()
    else if (wire === 5) this.advance(4)
    else throw new Error(`unsupported protobuf wire type ${wire}`)
  }
}

/** Calls onField for every field of a message; fields it does not consume (returns false) are skipped. */
function readFields(buf: Uint8Array, onField: (field: number, wire: number, r: Reader) => boolean): void {
  const r = new Reader(buf)
  while (!r.done) {
    const key = r.varint()
    const wire = key % 8
    if (!onField(Math.floor(key / 8), wire, r)) r.skip(wire)
  }
}

function readTensor(buf: Uint8Array): Tensor {
  const t: Tensor = { name: '', dims: [], dataType: 0, raw: new Uint8Array(0) }
  readFields(buf, (field, wire, r) => {
    if (field === 1 && wire === 0) t.dims.push(r.varint())
    else if (field === 1 && wire === 2) {
      const packed = new Reader(r.bytes())
      while (!packed.done) t.dims.push(packed.varint())
    } else if (field === 2 && wire === 0) t.dataType = r.varint()
    else if (field === 8 && wire === 2) t.name = new TextDecoder().decode(r.bytes())
    else if (field === 9 && wire === 2) t.raw = r.bytes()
    else return false
    return true
  })
  return t
}

/** ModelProto.graph (7) → GraphProto.initializer (5) → TensorProto, keyed by name. */
export function readInitializers(buf: ArrayBuffer): Map<string, Tensor> {
  const out = new Map<string, Tensor>()
  readFields(new Uint8Array(buf), (field, wire, r) => {
    if (field !== 7 || wire !== 2) return false
    readFields(r.bytes(), (gField, gWire, g) => {
      if (gField !== 5 || gWire !== 2) return false
      const t = readTensor(g.bytes())
      out.set(t.name, t)
      return true
    })
    return true
  })
  return out
}

/** Little-endian float32 values from bytes at any offset (raw_data is not 4-byte aligned in general). */
export function f32FromRaw(raw: Uint8Array): Float32Array {
  if (raw.byteLength % 4) throw new Error('float32 data length is not a multiple of 4')
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength)
  const out = new Float32Array(raw.byteLength / 4)
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true)
  return out
}
```

- [ ] **Step 6: Run tests and type check**

Run: `npx vitest run test/onnx.test.ts && npx tsc`
Expected: 4 tests PASS; `tsc` prints nothing.

- [ ] **Step 7: Commit**

```bash
git add word2vec/scripts/make_web_fixture.py word2vec/web/.gitignore word2vec/web/package.json word2vec/web/package-lock.json \
  word2vec/web/tsconfig.json word2vec/web/vite.config.ts word2vec/web/src/onnx.ts word2vec/web/test
git commit -m "feat(w2v-web): scaffold and ONNX initializer reader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Model, run names and downloads

**Files:**
- Create: `word2vec/web/src/model.ts`
- Test: `word2vec/web/test/model.test.ts`

**Interfaces:**
- Consumes: `readInitializers`, `f32FromRaw` from `src/onnx.ts`.
- Produces:
  - `type Lang = 'en' | 'cs' | 'uk'`
  - `interface RunSpec { lang: Lang; jepa: boolean; causal: boolean; dim: number; window: number }`
  - `HF_BASE: string`, `VOCAB_SIZE = 200_000`, `LARGE_MB = 20`
  - `runName(s: RunSpec): string`, `estimateMB(dim: number): number`
  - `class Model { words: string[]; dim: number; q: Int8Array; scales: Float32Array; index: Map<string, number>; invNorm: Float32Array; special: Uint8Array; constructor(words, dim, q, scales); static fromOnnx(words: string[], buf: ArrayBuffer): Model; id(word: string): number | undefined; unit(i: number): Float32Array }`
  - `type Progress = (what: string, loaded: number, total: number) => void`
  - `fetchBytes(url: string, what: string, onProgress: Progress): Promise<ArrayBuffer>`
  - `loadModel(spec: RunSpec, onProgress: Progress): Promise<Model>`

- [ ] **Step 1: Write the failing test**

`word2vec/web/test/model.test.ts`:

```ts
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/model.test.ts`
Expected: FAIL, cannot resolve `../src/model`.

- [ ] **Step 3: Implement `src/model.ts`**

```ts
import { f32FromRaw, readInitializers } from './onnx'

export type Lang = 'en' | 'cs' | 'uk'

export interface RunSpec {
  lang: Lang
  jepa: boolean
  causal: boolean
  dim: number
  window: number
}

export const HF_BASE = 'https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk/resolve/main/'
export const VOCAB_SIZE = 200_000
export const LARGE_MB = 20

export function runName(s: RunSpec): string {
  return `${s.lang}-${s.causal ? 'causal' : 'bidir'}${s.jepa ? '-jepa' : ''}-d${s.dim}-w${s.window}-s0`
}

/** Estimated model_int8.onnx size: one int8 per component plus a float32 scale per row. */
export function estimateMB(dim: number): number {
  return (VOCAB_SIZE * (dim + 4)) / 1e6
}

export class Model {
  readonly index = new Map<string, number>()
  /** 1 / L2 norm of each dequantized row (0 for all-zero rows). */
  readonly invNorm: Float32Array
  /** 1 for special tokens such as <pad>, which are never neighbours. */
  readonly special: Uint8Array

  constructor(
    readonly words: string[],
    readonly dim: number,
    readonly q: Int8Array,
    readonly scales: Float32Array,
  ) {
    const V = words.length
    if (q.length !== V * dim || scales.length !== V) throw new Error(`table shape does not match ${V} words × ${dim}`)
    this.invNorm = new Float32Array(V)
    this.special = new Uint8Array(V)
    for (let i = 0; i < V; i++) {
      const w = words[i]
      this.index.set(w, i)
      if (w.startsWith('<') && w.endsWith('>')) this.special[i] = 1
      let s = 0
      for (let j = i * dim, end = j + dim; j < end; j++) s += q[j] * q[j]
      const n = Math.sqrt(s) * scales[i]
      this.invNorm[i] = n > 0 ? 1 / n : 0
    }
  }

  static fromOnnx(words: string[], buf: ArrayBuffer): Model {
    const t = readInitializers(buf)
    const e = t.get('embeddings_int8')
    const s = t.get('row_scales')
    if (!e || !s) throw new Error('ONNX file has no embeddings_int8 / row_scales initializers')
    const [V, dim] = e.dims
    if (V !== words.length) throw new Error(`vocab has ${words.length} words but the table has ${V} rows`)
    return new Model(words, dim, new Int8Array(e.raw.buffer, e.raw.byteOffset, e.raw.byteLength), f32FromRaw(s.raw))
  }

  id(word: string): number | undefined {
    return this.index.get(word.toLowerCase())
  }

  unit(i: number): Float32Array {
    const out = new Float32Array(this.dim)
    const f = this.scales[i] * this.invNorm[i]
    const o = i * this.dim
    for (let j = 0; j < this.dim; j++) out[j] = this.q[o + j] * f
    return out
  }
}

export type Progress = (what: string, loaded: number, total: number) => void

export async function fetchBytes(url: string, what: string, onProgress: Progress): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`)
  const total = Number(res.headers.get('content-length')) || 0
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress(what, loaded, total)
  }
  const out = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) {
    out.set(c, off)
    off += c.length
  }
  return out.buffer
}

const vocabs = new Map<Lang, Promise<string[]>>()

function loadVocab(lang: Lang, onProgress: Progress): Promise<string[]> {
  let p = vocabs.get(lang)
  if (!p) {
    p = fetchBytes(`${HF_BASE}vocab/${lang}.json`, `${lang} vocabulary`, onProgress).then(
      (b) => (JSON.parse(new TextDecoder().decode(b)) as { words: string[] }).words,
    )
    vocabs.set(lang, p)
    p.catch(() => vocabs.delete(lang)) // retry on the next load instead of caching the failure
  }
  return p
}

export async function loadModel(spec: RunSpec, onProgress: Progress): Promise<Model> {
  const words = await loadVocab(spec.lang, onProgress)
  const name = runName(spec)
  const buf = await fetchBytes(`${HF_BASE}${name}/model_int8.onnx`, name, onProgress)
  return Model.fromOnnx(words, buf)
}
```

- [ ] **Step 4: Run tests and type check**

Run: `npx vitest run && npx tsc`
Expected: all tests PASS (onnx + model); `tsc` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add word2vec/web/src/model.ts word2vec/web/test/model.test.ts
git commit -m "feat(w2v-web): int8 model with row norms, run names and streamed downloads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Expression parser and word lists

**Files:**
- Create: `word2vec/web/src/expr.ts`
- Create: `word2vec/web/test/helpers.ts`
- Test: `word2vec/web/test/expr.test.ts`

**Interfaces:**
- Consumes: `Model` (`id`, `unit`, `dim`) from `src/model.ts`.
- Produces:
  - `interface Term { word: string; coef: number }`
  - `class ExprError extends Error`
  - `parse(src: string): Term[]` — words lowercased; throws `ExprError`.
  - `interface Resolved { terms: (Term & { id: number })[]; vector: Float32Array }`
  - `evaluate(model: Model, src: string): Resolved` — `vector = Σ coef · unit(word)`; throws `ExprError` on unknown words.
  - `splitWords(src: string): string[]` — split on whitespace/commas, lowercased, deduplicated, order kept.
  - test helper `makeModel(words: string[], rows: number[][], scale?: number): Model`

- [ ] **Step 1: Write the test helper and the failing test**

`word2vec/web/test/helpers.ts`:

```ts
import { Model } from '../src/model'

/** A Model from integer rows with one shared scale (rows[i] are the int8 values). */
export function makeModel(words: string[], rows: number[][], scale = 0.1): Model {
  return new Model(words, rows[0].length, Int8Array.from(rows.flat()), new Float32Array(words.length).fill(scale))
}
```

`word2vec/web/test/expr.test.ts`:

```ts
import { describe, expect, test } from 'vitest'
import { evaluate, ExprError, parse, splitWords } from '../src/expr'
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

describe('splitWords', () => {
  test('splits on commas and whitespace, lowercases and deduplicates', () => {
    expect(splitWords('King, queen  king\nprince,')).toEqual(['king', 'queen', 'prince'])
    expect(splitWords('  ')).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run test/expr.test.ts`
Expected: FAIL, cannot resolve `../src/expr`.

- [ ] **Step 3: Implement `src/expr.ts`**

```ts
import type { Model } from './model'

export interface Term {
  word: string
  coef: number
}

export interface Resolved {
  terms: (Term & { id: number })[]
  vector: Float32Array
}

export class ExprError extends Error {}

// [sign] [number [*]] word — a word cannot start with a digit, '.', '-', '+' or '*'
const TERM = /\s*([+-])?\s*(?:(\d+(?:\.\d+)?|\.\d+)\s*\*?\s*)?([^\s+*\d.\-][^\s+*]*)\s*/y

export function parse(src: string): Term[] {
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

export function evaluate(model: Model, src: string): Resolved {
  const terms = parse(src).map((t) => {
    const id = model.id(t.word)
    if (id === undefined) {
      const hint = t.word.includes('-') ? ' (put spaces around - to subtract)' : ''
      throw new ExprError(`unknown word “${t.word}”${hint}`)
    }
    return { ...t, id }
  })
  const vector = new Float32Array(model.dim)
  for (const t of terms) {
    const u = model.unit(t.id)
    for (let j = 0; j < model.dim; j++) vector[j] += t.coef * u[j]
  }
  return { terms, vector }
}

export function splitWords(src: string): string[] {
  return [...new Set(src.toLowerCase().split(/[\s,]+/).filter(Boolean))]
}
```

- [ ] **Step 4: Run tests and type check**

Run: `npx vitest run && npx tsc`
Expected: all tests PASS; `tsc` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add word2vec/web/src/expr.ts word2vec/web/test/helpers.ts word2vec/web/test/expr.test.ts
git commit -m "feat(w2v-web): word arithmetic parser and word-list splitting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: k-NN, PCA and axis projection (+ live check against the Hub)

**Files:**
- Create: `word2vec/web/src/knn.ts`, `word2vec/web/src/geom.ts`
- Test: `word2vec/web/test/knn.test.ts`, `word2vec/web/test/geom.test.ts`, `word2vec/web/test/live.test.ts`

**Interfaces:**
- Consumes: `Model` (`q`, `dim`, `scales`, `invNorm`, `special`, `words`, `unit`), `evaluate`, `loadModel`.
- Produces:
  - `interface Neighbour { id: number; word: string; score: number }`
  - `knn(model: Model, query: Float32Array, k: number, exclude?: ReadonlySet<number>): Neighbour[]` — descending cosine; `[]` for a zero query.
  - `dot(a: ArrayLike<number>, b: ArrayLike<number>): number`
  - `normalize(v: ArrayLike<number>): Float32Array` — copy; zero stays zero.
  - `pca2(vs: Float32Array[]): [number, number][]`
  - `projectAxes(vs: Float32Array[], x: Float32Array, y: Float32Array, orthogonalize: boolean): [number, number][]` — throws `Error` (`zero` / `parallel`).

- [ ] **Step 1: Write the failing tests**

`word2vec/web/test/knn.test.ts`:

```ts
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
```

`word2vec/web/test/geom.test.ts`:

```ts
import { describe, expect, test } from 'vitest'
import { pca2, projectAxes } from '../src/geom'

describe('pca2', () => {
  test('first component follows the dominant direction', () => {
    const t = [-2, -1, 0, 1, 2], z = [0.1, -0.1, 0.1, -0.1, 0]
    const vs = t.map((ti, i) => Float32Array.of(0.6 * ti, 0.8 * ti, z[i]))
    const c = pca2(vs)
    const s = Math.sign(c[4][0])
    c.forEach((p, i) => {
      expect(p[0]).toBeCloseTo(s * t[i], 5)
      expect(Math.abs(p[1])).toBeCloseTo(Math.abs(z[i]), 5)
    })
  })

  test('identical points collapse to the origin', () => {
    expect(pca2([Float32Array.of(1, 2), Float32Array.of(1, 2)])).toEqual([[0, 0], [0, 0]])
  })
})

describe('projectAxes', () => {
  const x = Float32Array.of(1, 0, 0), y = Float32Array.of(1, 1, 0)

  test('orthogonalizes y against x by default', () => {
    const [p] = projectAxes([Float32Array.of(1, 1, 0)], x, y, true)
    expect(p[0]).toBeCloseTo(Math.SQRT1_2, 6)
    expect(p[1]).toBeCloseTo(Math.SQRT1_2, 6)
  })

  test('uses raw y when asked', () => {
    const [p] = projectAxes([Float32Array.of(1, 1, 0)], x, y, false)
    expect(p[0]).toBeCloseTo(Math.SQRT1_2, 6)
    expect(p[1]).toBeCloseTo(1, 6)
  })

  test('an axis word lands at 1 on its own axis', () => {
    expect(projectAxes([Float32Array.of(3, 0, 0)], x, y, true)[0][0]).toBeCloseTo(1, 6)
  })

  test('rejects a zero axis and a y parallel to x', () => {
    expect(() => projectAxes([], new Float32Array(3), y, true)).toThrow(/zero/)
    expect(() => projectAxes([], x, Float32Array.of(2, 0, 0), true)).toThrow(/parallel/)
    expect(() => projectAxes([], x, Float32Array.of(2, 0, 0), false)).not.toThrow()
  })
})
```

`word2vec/web/test/live.test.ts` (downloads ~64 MB; runs only with `LIVE=1`):

```ts
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run test/knn.test.ts test/geom.test.ts`
Expected: FAIL, cannot resolve `../src/knn` and `../src/geom`.

- [ ] **Step 3: Implement `src/knn.ts`**

```ts
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
```

- [ ] **Step 4: Implement `src/geom.ts`**

```ts
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let j = 0; j < a.length; j++) s += a[j] * b[j]
  return s
}

export function normalize(v: ArrayLike<number>): Float32Array {
  const n = Math.sqrt(dot(v, v))
  return Float32Array.from(v, (x) => (n > 0 ? x / n : 0))
}

/** Coordinates of the centred points on their first two principal components (power iteration). */
export function pca2(vs: Float32Array[]): [number, number][] {
  const n = vs.length
  if (n === 0) return []
  const d = vs[0].length
  const mean = new Float64Array(d)
  for (const v of vs) for (let j = 0; j < d; j++) mean[j] += v[j] / n
  const X = vs.map((v) => Float64Array.from(v, (x, j) => x - mean[j]))
  const comps: Float64Array[] = []
  for (let c = 0; c < 2; c++) {
    let u = Float64Array.from({ length: d }, (_, j) => Math.sin(j + 1 + c)) // deterministic start
    for (let it = 0; it < 200; it++) {
      const w = new Float64Array(d)
      for (const x of X) {
        const a = dot(x, u)
        for (let j = 0; j < d; j++) w[j] += a * x[j]
      }
      for (const p of comps) {
        const a = dot(w, p)
        for (let j = 0; j < d; j++) w[j] -= a * p[j]
      }
      const nw = Math.sqrt(dot(w, w))
      if (nw < 1e-12) {
        u = new Float64Array(d)
        break
      }
      u = w.map((x) => x / nw)
    }
    comps.push(u)
  }
  return X.map((x) => [dot(x, comps[0]), dot(x, comps[1])])
}

/** (⟨v̂, x̂⟩, ⟨v̂, ŷ⟩) for each vector; ŷ is made orthogonal to x̂ first when asked. */
export function projectAxes(vs: Float32Array[], x: Float32Array, y: Float32Array, orthogonalize: boolean): [number, number][] {
  const xh = normalize(x)
  if (dot(xh, xh) === 0) throw new Error('x axis is the zero vector')
  const yNorm = Math.sqrt(dot(y, y))
  if (yNorm === 0) throw new Error('y axis is the zero vector')
  const yv = Float32Array.from(y)
  if (orthogonalize) {
    const a = dot(yv, xh)
    for (let j = 0; j < yv.length; j++) yv[j] -= a * xh[j]
    if (Math.sqrt(dot(yv, yv)) < 1e-6 * yNorm) throw new Error('y axis is parallel to x; untick “make y ⊥ x” or pick another y')
  }
  const yh = normalize(yv)
  return vs.map((v) => {
    const u = normalize(v)
    return [dot(u, xh), dot(u, yh)]
  })
}
```

- [ ] **Step 5: Run tests and type check**

Run: `npx vitest run && npx tsc`
Expected: all tests PASS, `live.test.ts` reported as skipped; `tsc` prints nothing.

- [ ] **Step 6: Run the live check against the Hub**

Run: `LIVE=1 npx vitest run test/live.test.ts`
Expected: PASS; the log shows the k-NN time (target: well under 1000 ms) and a neighbour list containing `queen`. If it fails, stop and report the printed neighbours. Do not loosen the assertion.

- [ ] **Step 7: Commit**

```bash
git add word2vec/web/src/knn.ts word2vec/web/src/geom.ts word2vec/web/test/knn.test.ts word2vec/web/test/geom.test.ts word2vec/web/test/live.test.ts
git commit -m "feat(w2v-web): cosine k-NN over the int8 table, PCA and axis projection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Page, SVG plot and wiring

**Files:**
- Create: `word2vec/web/index.html`, `word2vec/web/src/style.css`, `word2vec/web/src/plot.ts`, `word2vec/web/src/main.ts`

**Interfaces:**
- Consumes: everything from Tasks 1–4 (`loadModel`, `runName`, `estimateMB`, `LARGE_MB`, `Model`, `Lang`, `RunSpec`, `evaluate`, `ExprError`, `splitWords`, `Resolved`, `knn`, `normalize`, `pca2`, `projectAxes`).
- Produces:
  - `type Kind = 'term' | 'path' | 'result' | 'neighbour' | 'word'`
  - `interface Point { x: number; y: number; label: string; kind: Kind; onClick?: () => void }`
  - `renderPlot(svg: SVGSVGElement, points: Point[], opts?: { arrows?: [number, number][]; xLabel?: string; yLabel?: string; originLines?: boolean }): void`

No unit tests (DOM code); verified by `npm run build` and a manual run.

- [ ] **Step 1: Write `index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>word2vec explorer</title>
  </head>
  <body>
    <header>
      <h1>word2vec explorer</h1>
      <p>
        Word vectors trained on Wikipedia in English, Czech and Ukrainian
        (<a href="https://huggingface.co/mkkhavil/word2vec-wiki-cs-en-uk">models on Hugging Face</a>).
        Czech and Ukrainian words are lemmas. Everything runs in your browser.
      </p>
    </header>

    <section id="picker">
      <div class="row">
        <label>Language
          <select id="lang"><option value="en">English</option><option value="cs">Czech</option><option value="uk">Ukrainian</option></select>
        </label>
        <label>Objective
          <select id="objective"><option value="ns">negative sampling</option><option value="jepa">JEPA</option></select>
        </label>
        <label>Context
          <select id="direction"><option value="bidir">bidirectional</option><option value="causal">causal</option></select>
        </label>
        <label>Dim
          <select id="dim"><option>16</option><option>64</option><option>256</option><option selected>300</option></select>
        </label>
        <label>Window
          <select id="window"><option selected>5</option><option>10</option><option>20</option></select>
        </label>
        <button id="load">Load</button>
        <span id="size" class="status"></span>
        <span id="size-warn" role="img" aria-label="Large download" hidden>⚠</span>
      </div>
      <div class="row">
        <progress id="progress" hidden></progress>
        <span id="status" class="status">No model loaded.</span>
      </div>
    </section>

    <fieldset id="editor" disabled>
      <legend>Vector arithmetic</legend>
      <div class="row">
        <input id="expr" type="text" spellcheck="false" aria-label="Expression" />
        <label>k <input id="k" type="number" min="1" max="100" value="10" /></label>
        <button id="go">Go</button>
      </div>
      <p class="hint">
        Use + and − with spaces and optional weights: <code>king - man + woman</code>,
        <code>0.5*paris + berlin</code>. Click a neighbour to add it.
      </p>
      <p id="expr-error" class="error" role="alert"></p>
      <div class="split">
        <ol id="neighbours"></ol>
        <div>
          <div class="legend">
            <span><i class="sw term"></i>input words</span>
            <span><i class="sw result"></i>result</span>
            <span><i class="sw neighbour"></i>neighbours</span>
            <span><i class="sw path"></i>partial sums</span>
          </div>
          <svg id="pca-plot" class="plot" role="img" aria-label="PCA map of the input words, the result and its neighbours"></svg>
        </div>
      </div>
    </fieldset>

    <fieldset id="projection" disabled>
      <legend>2D projection</legend>
      <div class="row">
        <label for="xaxis">x axis</label>
        <input id="xaxis" type="text" spellcheck="false" />
      </div>
      <div class="row">
        <label for="yaxis">y axis</label>
        <input id="yaxis" type="text" spellcheck="false" />
        <label><input id="ortho" type="checkbox" checked /> make y ⊥ x</label>
      </div>
      <div class="row">
        <textarea id="words" rows="2" spellcheck="false" aria-label="Words to plot"></textarea>
        <button id="plot">Plot</button>
      </div>
      <p id="proj-error" class="error" role="alert"></p>
      <svg id="proj-plot" class="plot" role="img" aria-label="Words projected onto the chosen axes"></svg>
    </fieldset>

    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 2: Write `src/style.css`**

```css
:root {
  color-scheme: light dark;
  --surface: #fcfcfb;
  --panel: #f3f2ef;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --rule: #dcdbd6;
  --term: #2a78d6;
  --result: #eb6834;
  --neighbour: #1baf7a;
  font: 15px/1.45 system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19;
    --panel: #232321;
    --ink: #f0efec;
    --ink-2: #c3c2b7;
    --rule: #3a3a37;
    --term: #3987e5;
    --result: #d95926;
    --neighbour: #199e70;
  }
}
body { max-width: 1100px; margin: 0 auto; padding: 16px; background: var(--surface); color: var(--ink); }
a { color: var(--term); }
header p, .hint, .status { color: var(--ink-2); }
section, fieldset { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 16px; margin: 0 0 16px; }
fieldset:disabled { opacity: 0.55; }
legend { font-weight: 600; }
.row { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 8px; }
input[type='text'], textarea { flex: 1 1 260px; font: inherit; padding: 4px 6px; }
input[type='number'] { width: 4em; }
.error { min-height: 1.4em; margin: 4px 0; }
.error:not(:empty)::before { content: '⚠ '; }
#size-warn { cursor: help; }
progress { width: 200px; }
.split { display: grid; grid-template-columns: minmax(180px, 1fr) 3fr; gap: 16px; }
@media (max-width: 720px) { .split { grid-template-columns: 1fr; } }
#neighbours { margin: 0; padding-left: 2em; font-variant-numeric: tabular-nums; }
#neighbours button { all: unset; cursor: pointer; text-decoration: underline dotted; }
#neighbours .score { color: var(--ink-2); margin-left: 0.5em; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--ink-2); margin-bottom: 4px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
.sw.term { background: var(--term); }
.sw.result { background: var(--result); }
.sw.neighbour { background: var(--neighbour); }
.sw.path { background: var(--ink-2); }
svg.plot { display: block; width: 100%; height: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
.plot .axis { stroke: var(--rule); stroke-width: 1; }
.plot .arrow { stroke: var(--ink-2); stroke-width: 2; }
.plot .arrowhead { fill: var(--ink-2); }
.plot .hit { fill: transparent; }
.plot .dot { stroke: var(--surface); stroke-width: 2; }
.plot .term .dot, .plot .word .dot { fill: var(--term); }
.plot .result .dot { fill: var(--result); }
.plot .neighbour .dot { fill: var(--neighbour); }
.plot .path .dot { fill: var(--ink-2); }
.plot .label { fill: var(--ink); font-size: 12px; paint-order: stroke; stroke: var(--surface); stroke-width: 3px; }
.plot .neighbour .label, .plot .path .label { fill: var(--ink-2); }
.plot .axis-title { fill: var(--ink-2); font-size: 12px; }
.plot .clickable { cursor: pointer; }
```

- [ ] **Step 3: Write `src/plot.ts`**

```ts
const SVG_NS = 'http://www.w3.org/2000/svg'
const W = 640
const H = 440
const PAD = 40

export type Kind = 'term' | 'path' | 'result' | 'neighbour' | 'word'

export interface Point {
  x: number
  y: number
  label: string
  kind: Kind
  onClick?: () => void
}

export interface PlotOptions {
  /** Pairs of point indices, drawn as arrows from the first to the second. */
  arrows?: [number, number][]
  xLabel?: string
  yLabel?: string
  /** Draw the x = 0 and y = 0 lines (and keep the origin in view). */
  originLines?: boolean
}

const DRAW_ORDER: Kind[] = ['neighbour', 'word', 'path', 'term', 'result']

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag)
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v))
  parent.appendChild(e)
  return e
}

function extent(v: number[]): [number, number] {
  let lo = Math.min(...v)
  let hi = Math.max(...v)
  if (hi - lo < 1e-9) {
    lo -= 1
    hi += 1
  }
  const m = (hi - lo) * 0.08
  return [lo - m, hi + m]
}

export function renderPlot(svg: SVGSVGElement, points: Point[], opts: PlotOptions = {}): void {
  svg.replaceChildren()
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`)
  if (points.length === 0) return
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  if (opts.originLines) {
    xs.push(0)
    ys.push(0)
  }
  const [x0, x1] = extent(xs)
  const [y0, y1] = extent(ys)
  const sx = (x: number) => PAD + ((x - x0) / (x1 - x0)) * (W - 2 * PAD)
  const sy = (y: number) => H - PAD - ((y - y0) / (y1 - y0)) * (H - 2 * PAD)

  const headId = `${svg.id}-head`
  const marker = el('marker', { id: headId, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, svg))
  el('path', { d: 'M0,0 L10,5 L0,10 z', class: 'arrowhead' }, marker)

  if (opts.originLines) {
    el('line', { x1: sx(0), y1: PAD / 2, x2: sx(0), y2: H - PAD / 2, class: 'axis' }, svg)
    el('line', { x1: PAD / 2, y1: sy(0), x2: W - PAD / 2, y2: sy(0), class: 'axis' }, svg)
  }

  for (const [a, b] of opts.arrows ?? []) {
    const ax = sx(points[a].x), ay = sy(points[a].y), bx = sx(points[b].x), by = sy(points[b].y)
    const len = Math.hypot(bx - ax, by - ay)
    if (len < 16) continue
    const s = 6 / len, e = (len - 8) / len // stop short of the dots
    el('line', { x1: ax + (bx - ax) * s, y1: ay + (by - ay) * s, x2: ax + (bx - ax) * e, y2: ay + (by - ay) * e, class: 'arrow', 'marker-end': `url(#${headId})` }, svg)
  }

  const sorted = [...points].sort((p, q) => DRAW_ORDER.indexOf(p.kind) - DRAW_ORDER.indexOf(q.kind))
  for (const p of sorted) {
    const g = el('g', { class: `pt ${p.kind}`, transform: `translate(${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)})` }, svg)
    el('title', {}, g).textContent = `${p.label}  (${p.x.toFixed(3)}, ${p.y.toFixed(3)})`
    el('circle', { r: 12, class: 'hit' }, g)
    el('circle', { r: p.kind === 'result' ? 6 : 4, class: 'dot' }, g)
    el('text', { x: 8, y: -8, class: 'label' }, g).textContent = p.label
    if (p.onClick) {
      g.classList.add('clickable')
      g.addEventListener('click', p.onClick)
    }
  }

  if (opts.xLabel) el('text', { x: W / 2, y: H - 8, class: 'axis-title', 'text-anchor': 'middle' }, svg).textContent = `x: ${opts.xLabel}`
  if (opts.yLabel) {
    el('text', { x: 14, y: H / 2, class: 'axis-title', 'text-anchor': 'middle', transform: `rotate(-90 14 ${H / 2})` }, svg).textContent = `y: ${opts.yLabel}`
  }
}
```

- [ ] **Step 4: Write `src/main.ts`**

```ts
import { evaluate, ExprError, splitWords, type Resolved } from './expr'
import { normalize, pca2, projectAxes } from './geom'
import { knn } from './knn'
import { estimateMB, LARGE_MB, loadModel, runName, type Lang, type Model, type RunSpec } from './model'
import { renderPlot, type Point } from './plot'
import './style.css'

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T
const value = (id: string) => $<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(id).value

const DEFAULTS: Record<Lang, { expr: string; x: string; y: string; words: string }> = {
  en: { expr: 'king - man + woman', x: 'woman - man', y: 'rich - poor', words: 'king queen prince princess doctor nurse engineer teacher beggar billionaire' },
  cs: { expr: 'král - muž + žena', x: 'žena - muž', y: 'bohatý - chudý', words: 'král královna princ princezna lékař sestra učitel inženýr žebrák milionář' },
  uk: { expr: 'король - чоловік + жінка', x: 'жінка - чоловік', y: 'багатий - бідний', words: 'король королева принц принцеса лікар медсестра вчитель інженер жебрак мільйонер' },
}

let model: Model | null = null
let modelLang: Lang | null = null

function currentSpec(): RunSpec {
  return {
    lang: value('lang') as Lang,
    jepa: value('objective') === 'jepa',
    causal: value('direction') === 'causal',
    dim: Number(value('dim')),
    window: Number(value('window')),
  }
}

function updateSize(): void {
  const mb = estimateMB(currentSpec().dim)
  const n = Math.round(mb)
  $('size').textContent = `≈ ${n} MB`
  const warn = $('size-warn')
  warn.hidden = mb <= LARGE_MB
  warn.title = `Large download (≈${n} MB), so consider a smaller dim on mobile data.`
}

function fillDefaults(lang: Lang): void {
  const d = DEFAULTS[lang]
  $<HTMLInputElement>('expr').value = d.expr
  $<HTMLInputElement>('xaxis').value = d.x
  $<HTMLInputElement>('yaxis').value = d.y
  $<HTMLTextAreaElement>('words').value = d.words
}

async function onLoad(): Promise<void> {
  const spec = currentSpec()
  const name = runName(spec)
  const button = $<HTMLButtonElement>('load')
  const status = $('status')
  const bar = $<HTMLProgressElement>('progress')
  button.disabled = true
  bar.hidden = false
  bar.removeAttribute('value')
  try {
    model = await loadModel(spec, (what, loaded, total) => {
      const mb = (b: number) => (b / 1e6).toFixed(1)
      status.textContent = `Downloading ${what}: ${mb(loaded)}${total ? ` / ${mb(total)}` : ''} MB`
      if (total) {
        bar.max = total
        bar.value = loaded
      }
    })
    status.textContent = `Loaded ${name}: ${model.words.length.toLocaleString()} words, ${model.dim} dimensions.`
    if (modelLang !== spec.lang) {
      fillDefaults(spec.lang)
      modelLang = spec.lang
    }
    $<HTMLFieldSetElement>('editor').disabled = false
    $<HTMLFieldSetElement>('projection').disabled = false
    runEditor()
    runProjection()
  } catch (e) {
    status.textContent = `Could not load ${name}: ${(e as Error).message}${model ? ' The previous model is still loaded.' : ''}`
  } finally {
    button.disabled = false
    bar.hidden = true
  }
}

function formatPrefix(terms: Resolved['terms'], n: number): string {
  return terms
    .slice(0, n)
    .map((t, i) => {
      const c = Math.abs(t.coef)
      const w = c === 1 ? t.word : `${c}·${t.word}`
      return i === 0 ? (t.coef < 0 ? `−${w}` : w) : `${t.coef < 0 ? '−' : '+'} ${w}`
    })
    .join(' ')
}

function addTerm(word: string): void {
  const e = $<HTMLInputElement>('expr')
  e.value = `${e.value.trim()} + ${word}`
  runEditor()
}

function runEditor(): void {
  const m = model
  if (!m) return
  const err = $('expr-error')
  const list = $('neighbours')
  const svg = $<SVGSVGElement>('pca-plot')
  err.textContent = ''
  let res: Resolved
  try {
    res = evaluate(m, value('expr'))
  } catch (e) {
    if (!(e instanceof ExprError)) throw e
    err.textContent = e.message
    return
  }
  const result = normalize(res.vector)
  if (result.every((x) => x === 0)) {
    err.textContent = 'The expression sums to the zero vector.'
    list.replaceChildren()
    renderPlot(svg, [])
    return
  }
  const k = Math.min(100, Math.max(1, Math.round(Number(value('k'))) || 10))
  const nbs = knn(m, res.vector, k, new Set(res.terms.map((t) => t.id)))

  list.replaceChildren(
    ...nbs.map((nb) => {
      const li = document.createElement('li')
      const b = document.createElement('button')
      const s = document.createElement('span')
      b.textContent = nb.word
      b.title = `add + ${nb.word}`
      b.addEventListener('click', () => addTerm(nb.word))
      s.className = 'score'
      s.textContent = nb.score.toFixed(3)
      li.append(b, s)
      return li
    }),
  )

  // PCA map of unit vectors: input words, partial sums, the result and its neighbours.
  const vecs: Float32Array[] = []
  const pts: Omit<Point, 'x' | 'y'>[] = []
  const add = (v: Float32Array, p: Omit<Point, 'x' | 'y'>) => {
    vecs.push(v)
    pts.push(p)
    return vecs.length - 1
  }
  const seen = new Set<number>()
  for (const t of res.terms) {
    if (seen.has(t.id)) continue
    seen.add(t.id)
    add(m.unit(t.id), { label: t.word, kind: 'term' })
  }
  const chain = [0] // first term → each partial sum → result
  const acc = new Float32Array(m.dim)
  res.terms.forEach((t, i) => {
    const u = m.unit(t.id)
    for (let j = 0; j < m.dim; j++) acc[j] += t.coef * u[j]
    if (i > 0 && i < res.terms.length - 1) chain.push(add(normalize(acc), { label: formatPrefix(res.terms, i + 1), kind: 'path' }))
  })
  const r = add(result, { label: 'result', kind: 'result' })
  if (res.terms.length > 1) chain.push(r)
  for (const nb of nbs) add(m.unit(nb.id), { label: nb.word, kind: 'neighbour', onClick: () => addTerm(nb.word) })

  const xy = pca2(vecs)
  const arrows = chain.slice(1).map((b, i) => [chain[i], b] as [number, number])
  renderPlot(svg, pts.map((p, i) => ({ ...p, x: xy[i][0], y: xy[i][1] })), { arrows })
}

function axis(m: Model, id: string, name: string): Float32Array {
  try {
    return evaluate(m, value(id)).vector
  } catch (e) {
    if (e instanceof ExprError) throw new ExprError(`${name}: ${e.message}`)
    throw e
  }
}

function runProjection(): void {
  const m = model
  if (!m) return
  const err = $('proj-error')
  const svg = $<SVGSVGElement>('proj-plot')
  const ortho = $<HTMLInputElement>('ortho').checked
  err.textContent = ''
  const known: { word: string; id: number }[] = []
  const unknown: string[] = []
  for (const w of splitWords(value('words'))) {
    const id = m.id(w)
    if (id === undefined) unknown.push(w)
    else known.push({ word: w, id })
  }
  let xy: [number, number][]
  try {
    xy = projectAxes(known.map((k) => m.unit(k.id)), axis(m, 'xaxis', 'x axis'), axis(m, 'yaxis', 'y axis'), ortho)
  } catch (e) {
    err.textContent = (e as Error).message
    renderPlot(svg, [])
    return
  }
  if (unknown.length) err.textContent = `Not in the vocabulary, skipped: ${unknown.join(', ')}`
  renderPlot(svg, known.map((k, i) => ({ x: xy[i][0], y: xy[i][1], label: k.word, kind: 'word' })), {
    originLines: true,
    xLabel: value('xaxis'),
    yLabel: value('yaxis') + (ortho ? ' (⊥ x)' : ''),
  })
}

for (const id of ['lang', 'objective', 'direction', 'dim', 'window']) $(id).addEventListener('change', updateSize)
$('load').addEventListener('click', onLoad)
$('go').addEventListener('click', runEditor)
$('expr').addEventListener('keydown', (e) => e.key === 'Enter' && runEditor())
$('k').addEventListener('change', runEditor)
$('plot').addEventListener('click', runProjection)
for (const id of ['xaxis', 'yaxis']) $(id).addEventListener('keydown', (e) => e.key === 'Enter' && runProjection())
$('ortho').addEventListener('change', runProjection)
fillDefaults('en')
updateSize()
```

- [ ] **Step 5: Build**

Run: `npm test && npm run build`
Expected: tests PASS; `tsc` clean; `dist/index.html` references `/datalab/assets/…`. Check with `grep -o '/datalab/assets/[^"]*' dist/index.html`.

- [ ] **Step 6: Manual check in a browser**

Run: `npx vite --port 5173` and open `http://localhost:5173/datalab/`. Check:
1. With the default dim 300, `≈ 61 MB` and the ⚠ show; its tooltip reads `Large download (≈61 MB), so consider a smaller dim on mobile data.` With dim 64 the ⚠ disappears (`≈ 14 MB`).
2. Load en d16: the progress bar moves, the status shows `Loaded en-bidir-d16-w5-s0: 200,000 words, 16 dimensions.`, and both panels enable and render.
3. Load en d300. `king - man + woman` lists `queen` near the top. The PCA map shows blue terms, a grey `king − man` partial sum, an orange result and aqua neighbours, with arrows king → king − man → result. Clicking a neighbour appends `+ word`.
4. Typing `king - king` gives the zero-vector message, and `woman-man` gives the unknown-word message with the spaces hint.
5. The default projection plots 10 words on axes through the origin. Setting y = `2*woman - 2*man` with x = `woman - man` and "make y ⊥ x" ticked gives the parallel error; unticking it plots.
6. Switch the OS or browser to dark mode: colours switch, and labels stay readable.
7. Load cs d64: the Czech defaults appear and plot.

If a browser is not available to the implementer, report that and leave this step for the user. Do not mark it done.

- [ ] **Step 7: Commit**

```bash
git add word2vec/web/index.html word2vec/web/src/style.css word2vec/web/src/plot.ts word2vec/web/src/main.ts
git commit -m "feat(w2v-web): explorer page with arithmetic, k-NN, PCA map and 2D projection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Pages deployment and README

**Files:**
- Create: `.github/workflows/pages.yml` (repo root `datalab/`)
- Modify: `word2vec/README.md` (append a `## Web explorer` section at the end)

**Interfaces:**
- Consumes: `npm ci`, `npm test`, `npm run build` scripts and `dist/` from Tasks 1–5.
- Produces: the deployed site at `https://zhigamason.github.io/datalab/`.

- [ ] **Step 1: Write the workflow**

`.github/workflows/pages.yml`:

```yaml
name: Deploy word2vec explorer to Pages

on:
  push:
    branches: [main]
    paths: ['word2vec/web/**', '.github/workflows/pages.yml']
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: word2vec/web
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v5
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: word2vec/web/package-lock.json
      - run: npm ci
      - run: npm test
      - run: npm run build
      - uses: actions/upload-pages-artifact@v4
        with:
          path: word2vec/web/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

Validate: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/pages.yml')); print('ok')"` (from the repo root).
Expected: `ok`.

- [ ] **Step 2: Append the README section**

Append to `word2vec/README.md`:

~~~markdown
## Web explorer

`web/` is a static page (<https://zhigamason.github.io/datalab/>) that downloads one int8 table
from the Hugging Face repo on demand and runs everything in the browser: word arithmetic with
k nearest neighbours and a PCA map, and a 2D projection onto two word/expression axes. It reads
`embeddings_int8` and `row_scales` straight from `model_int8.onnx` with a small protobuf reader,
so it needs no ONNX runtime. It has no runtime dependencies.

```bash
cd web
npm install
npm run dev                          # http://localhost:5173/datalab/
npm test                             # unit tests; LIVE=1 npm test also checks the published en d300 model
uv run python -m scripts.make_web_fixture   # from word2vec/, regenerates test/fixtures/tiny.onnx
```

`.github/workflows/pages.yml` builds and deploys it on every push to `main` that touches `word2vec/web/`.
One-time setup: in the GitHub repo settings, under Pages → Build and deployment → Source, choose **GitHub Actions**.
~~~

- [ ] **Step 3: Final check**

Run from `word2vec/web`: `npm ci && npm test && npm run build`
Expected: clean install, all tests PASS, build succeeds.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pages.yml word2vec/README.md
git commit -m "ci(w2v-web): deploy the explorer to GitHub Pages; README section

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

After merging to `main`, the user enables Pages → Source: GitHub Actions. The first deploy can also be started manually from the Actions tab (`workflow_dispatch`).
