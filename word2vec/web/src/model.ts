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
  try {
    return await download(url, what, onProgress)
  } catch (e) {
    const msg = (e as Error).message
    throw new Error(msg.includes(url) ? msg : `${msg} for ${url}`) // network errors only say "Failed to fetch"
  }
}

async function download(url: string, what: string, onProgress: Progress): Promise<ArrayBuffer> {
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
