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
