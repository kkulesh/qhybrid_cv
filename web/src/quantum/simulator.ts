// Dense state-vector simulator. Qubit q corresponds to bit q of the basis index (q[0] is the least
// significant bit), the same convention as OpenQASM/Qiskit bitstrings "q[n-1] ... q[0]".
import type { Circuit, Op } from './gates'

type C = [number, number] // complex number as [re, im]
type M2 = [C, C, C, C] // 2x2 matrix, row-major

const S2 = Math.SQRT1_2
const cexp = (phi: number): C => [Math.cos(phi), Math.sin(phi)]

function matrix(op: Op): M2 | null {
  const t = op.params[0] ?? 0
  const c = Math.cos(t / 2)
  const s = Math.sin(t / 2)
  switch (op.kind) {
    case 'H': return [[S2, 0], [S2, 0], [S2, 0], [-S2, 0]]
    case 'X': case 'CX': case 'CCX': return [[0, 0], [1, 0], [1, 0], [0, 0]]
    case 'Y': return [[0, 0], [0, -1], [0, 1], [0, 0]]
    case 'Z': case 'CZ': return [[1, 0], [0, 0], [0, 0], [-1, 0]]
    case 'S': return [[1, 0], [0, 0], [0, 0], [0, 1]]
    case 'SDG': return [[1, 0], [0, 0], [0, 0], [0, -1]]
    case 'T': return [[1, 0], [0, 0], [0, 0], cexp(Math.PI / 4)]
    case 'TDG': return [[1, 0], [0, 0], [0, 0], cexp(-Math.PI / 4)]
    case 'SX': return [[0.5, 0.5], [0.5, -0.5], [0.5, -0.5], [0.5, 0.5]]
    case 'RX': return [[c, 0], [0, -s], [0, -s], [c, 0]]
    case 'RY': return [[c, 0], [-s, 0], [s, 0], [c, 0]]
    case 'RZ': return [cexp(-t / 2), [0, 0], [0, 0], cexp(t / 2)]
    case 'P': return [[1, 0], [0, 0], [0, 0], cexp(t)]
    default: return null
  }
}

export class State {
  re: Float64Array
  im: Float64Array
  constructor(public n: number) {
    this.re = new Float64Array(1 << n)
    this.im = new Float64Array(1 << n)
    this.re[0] = 1
  }

  /** Apply a 2x2 matrix to `target`, only on basis states where all `controls` are 1. */
  apply1(m: M2, target: number, controls: number[] = []) {
    const tb = 1 << target
    const cmask = controls.reduce((a, q) => a | (1 << q), 0)
    const { re, im } = this
    for (let i = 0; i < re.length; i++) {
      if (i & tb || (i & cmask) !== cmask) continue
      const j = i | tb
      const ar = re[i], ai = im[i], br = re[j], bi = im[j]
      re[i] = m[0][0] * ar - m[0][1] * ai + m[1][0] * br - m[1][1] * bi
      im[i] = m[0][0] * ai + m[0][1] * ar + m[1][0] * bi + m[1][1] * br
      re[j] = m[2][0] * ar - m[2][1] * ai + m[3][0] * br - m[3][1] * bi
      im[j] = m[2][0] * ai + m[2][1] * ar + m[3][0] * bi + m[3][1] * br
    }
  }

  swap(a: number, b: number) {
    const ab = 1 << a, bb = 1 << b
    const { re, im } = this
    for (let i = 0; i < re.length; i++) {
      if (i & ab && !(i & bb)) {
        const j = (i & ~ab) | bb
        ;[re[i], re[j]] = [re[j], re[i]]
        ;[im[i], im[j]] = [im[j], im[i]]
      }
    }
  }

  /** exp(-i θ/2 Z_a Z_b): phase e^{-iθ/2} on even parity, e^{+iθ/2} on odd parity. */
  rzz(a: number, b: number, theta: number) {
    const { re, im } = this
    for (let i = 0; i < re.length; i++) {
      const odd = ((i >> a) ^ (i >> b)) & 1
      const [cr, ci] = cexp(odd ? theta / 2 : -theta / 2)
      const r = re[i], m = im[i]
      re[i] = r * cr - m * ci
      im[i] = r * ci + m * cr
    }
  }

  probabilities(): Float64Array {
    const p = new Float64Array(this.re.length)
    for (let i = 0; i < p.length; i++) p[i] = this.re[i] ** 2 + this.im[i] ** 2
    return p
  }

  /** Bloch vector of qubit q from its reduced density matrix. */
  bloch(q: number): [number, number, number] {
    const b = 1 << q
    let p0 = 0, p1 = 0, r01 = 0, i01 = 0
    for (let i = 0; i < this.re.length; i++) {
      if (i & b) continue
      const j = i | b
      const ar = this.re[i], ai = this.im[i], br = this.re[j], bi = this.im[j]
      p0 += ar * ar + ai * ai
      p1 += br * br + bi * bi
      r01 += ar * br + ai * bi // Re(a * conj(b))
      i01 += ai * br - ar * bi // Im(a * conj(b))
    }
    return [2 * r01, -2 * i01, p0 - p1]
  }
}

export interface SimResult {
  state: State
  probs: Float64Array
  measured: number[] // qubits that carry a measurement
  warnings: string[]
}

export function simulate(circuit: Circuit): SimResult {
  const st = new State(circuit.nQubits)
  const ops = [...circuit.ops].sort((a, b) => a.col - b.col)
  const measuredAt = new Map<number, number>()
  const warnings: string[] = []
  for (const op of ops) {
    if (op.kind === 'BARRIER') continue
    if (op.kind === 'MEASURE') {
      measuredAt.set(op.wires[0], op.col)
      continue
    }
    for (const w of op.wires) {
      if (measuredAt.has(w)) {
        warnings.push(`Вентиль ${op.kind} на q[${w}] стоїть після вимірювання; симулятор вважає вимірювання кінцевими.`)
      }
    }
    if (op.kind === 'SWAP') st.swap(op.wires[0], op.wires[1])
    else if (op.kind === 'RZZ') st.rzz(op.wires[0], op.wires[1], op.params[0])
    else {
      const m = matrix(op)
      if (m) st.apply1(m, op.wires[op.wires.length - 1], op.wires.slice(0, -1))
    }
  }
  return { state: st, probs: st.probabilities(), measured: [...measuredAt.keys()].sort((a, b) => a - b), warnings: [...new Set(warnings)] }
}

/** Sample `shots` outcomes of the measured qubits; keys are bitstrings over measured qubits, highest first. */
export function sampleCounts(probs: Float64Array, measured: number[], shots: number, seed = 1): Map<string, number> {
  const cdf = new Float64Array(probs.length)
  let acc = 0
  for (let i = 0; i < probs.length; i++) cdf[i] = acc += probs[i]
  let s = seed >>> 0 || 1
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * acc
  const counts = new Map<string, number>()
  const order = [...measured].sort((a, b) => b - a)
  for (let k = 0; k < shots; k++) {
    const r = rand()
    let lo = 0, hi = cdf.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (cdf[mid] < r) lo = mid + 1
      else hi = mid
    }
    const key = order.map((q) => ((lo >> q) & 1).toString()).join('')
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  return counts
}

export const bitstring = (i: number, n: number) => i.toString(2).padStart(n, '0')
