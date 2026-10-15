import { describe, expect, test } from 'vitest'
import { pca2, projectAxes } from '../src/geom'

describe('pca2', () => {
  test('first component follows the dominant direction', () => {
    const t = [-2, -1, 0, 1, 2], z = [0.1, -0.1, 0, -0.1, 0.1] // mean 0, uncorrelated with t
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
