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
