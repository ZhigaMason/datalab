import { axisVector, evaluate, ExprError, isZero, splitWords, ZERO_ID, type Resolved } from './expr'
import { normalize, pca2, projectAxes } from './geom'
import { knn } from './knn'
import { estimateMB, LARGE_MB, loadModel, runName, type Lang, type Model, type RunSpec } from './model'
import { renderPlot, type Point } from './plot'
import './style.css'

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T
const value = (id: string) => $<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(id).value

// x and y are [negative end, positive end]
const DEFAULTS: Record<Lang, { expr: string; x: [string, string]; y: [string, string]; words: string }> = {
  en: { expr: 'king - man + woman', x: ['man', 'woman'], y: ['poor', 'rich'], words: 'king queen prince princess doctor nurse engineer teacher beggar billionaire' },
  cs: { expr: 'král - muž + žena', x: ['muž', 'žena'], y: ['chudý', 'bohatý'], words: 'král královna princ princezna lékař sestra učitel inženýr žebrák milionář' },
  uk: { expr: 'король - чоловік + жінка', x: ['чоловік', 'жінка'], y: ['бідний', 'багатий'], words: 'король королева принц принцеса лікар медсестра вчитель інженер жебрак мільйонер' },
}

let model: Model | null = null

/** Unit vector of a vocab row, or the zero vector for the word `0`. */
const termVec = (m: Model, id: number) => (id === ZERO_ID ? new Float32Array(m.dim) : m.unit(id))
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
  ;[$<HTMLInputElement>('xaxis-neg').value, $<HTMLInputElement>('xaxis-pos').value] = d.x
  ;[$<HTMLInputElement>('yaxis-neg').value, $<HTMLInputElement>('yaxis-pos').value] = d.y
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
  if (isZero(res)) {
    err.textContent = 'The expression sums to the zero vector.'
    list.replaceChildren()
    renderPlot(svg, [])
    return
  }
  const result = normalize(res.vector)
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
    add(termVec(m, t.id), { label: t.word, kind: 'term' })
  }
  const chain = [0] // first term → each partial sum → result
  const acc = new Float32Array(m.dim)
  res.terms.forEach((t, i) => {
    const u = termVec(m, t.id)
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
    return axisVector(m, value(`${id}-neg`), value(`${id}-pos`))
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
    const id = w === '0' ? ZERO_ID : m.id(w)
    if (id === undefined) unknown.push(w)
    else known.push({ word: w, id })
  }
  let xy: [number, number][]
  try {
    xy = projectAxes(known.map((k) => termVec(m, k.id)), axis(m, 'xaxis', 'x axis'), axis(m, 'yaxis', 'y axis'), ortho)
  } catch (e) {
    err.textContent = (e as Error).message
    renderPlot(svg, [])
    return
  }
  if (unknown.length) err.textContent = `Not in the vocabulary, skipped: ${unknown.join(', ')}`
  renderPlot(svg, known.map((k, i) => ({ x: xy[i][0], y: xy[i][1], label: k.word, kind: 'word' })), {
    originLines: true,
    ends: { left: value('xaxis-neg'), right: value('xaxis-pos'), bottom: value('yaxis-neg'), top: value('yaxis-pos') },
    yLabel: ortho ? 'made ⊥ x' : undefined,
  })
}

for (const id of ['lang', 'objective', 'direction', 'dim', 'window']) $(id).addEventListener('change', updateSize)
$('load').addEventListener('click', onLoad)
$('go').addEventListener('click', runEditor)
$('expr').addEventListener('keydown', (e) => e.key === 'Enter' && runEditor())
$('k').addEventListener('change', runEditor)
$('plot').addEventListener('click', runProjection)
for (const id of ['xaxis-neg', 'xaxis-pos', 'yaxis-neg', 'yaxis-pos']) $(id).addEventListener('keydown', (e) => e.key === 'Enter' && runProjection())
$('ortho').addEventListener('change', runProjection)
fillDefaults('en')
updateSize()
