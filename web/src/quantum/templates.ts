// Ready-made circuits, including the encodings used in the QHybrid-CV experiments.
import { compact } from './layout'
import { newId, type Circuit, type GateKind } from './gates'

type Step = [GateKind, number[], number[]?]

function build(n: number, steps: Step[]): Circuit {
  const ops = steps.map(([kind, wires, params], col) => ({ id: newId(), kind, wires, params: params ?? [], col }))
  return { nQubits: n, ops: compact(ops, n) }
}

const measureAll = (n: number): Step[] => [...Array(n).keys()].map((q) => ['MEASURE', [q]])

// Example feature vector after the encoding-aware adapter (angle range [-π/4, π/4] for γ = 0.5).
const X = [0.62, -0.35, 0.18, -0.71]

/** StronglyEntanglingLayers: Rot(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ) on every qubit, then a CNOT ring
 *  with range r = (l mod (n-1)) + 1, as in PennyLane. */
function sel(n: number, layers: number): Step[] {
  const w = [0.3, 1.1, -0.4, 0.8, -1.2, 0.5, 0.9, -0.6, 0.2, -0.9, 1.3, 0.4]
  const steps: Step[] = []
  for (let l = 0; l < layers; l++) {
    for (let q = 0; q < n; q++) {
      const k = (l * n + q) % w.length
      steps.push(['RZ', [q], [w[k]]], ['RY', [q], [w[(k + 1) % w.length]]], ['RZ', [q], [w[(k + 2) % w.length]]])
    }
    const r = (l % (n - 1)) + 1
    for (let q = 0; q < n; q++) steps.push(['CX', [q, (q + r) % n]])
  }
  return steps
}

// Example of 8 adapter outputs for amplitude encoding on 3 qubits (normalised to a unit vector inside).
const A = [0.42, -0.18, 0.31, 0.05, -0.27, 0.55, 0.12, -0.38]

const popcount = (v: number) => { let c = 0; for (; v; v &= v - 1) c++; return c }
const gray = (i: number) => i ^ (i >> 1)

/** Uniformly controlled RY: angle theta[j] on `target` when the controls hold pattern j (bit m of j is
 *  ctrl[m]). Decomposed into 2^k RY and 2^k CNOT along a Gray code (Möttönen et al., 2004). */
function ucry(target: number, ctrl: number[], theta: number[]): Step[] {
  const k = ctrl.length, N = 1 << k
  if (k === 0) return [['RY', [target], [theta[0]]]]
  const steps: Step[] = []
  for (let i = 0; i < N; i++) {
    let a = 0
    for (let j = 0; j < N; j++) a += (popcount(j & gray(i)) % 2 ? -1 : 1) * theta[j]
    steps.push(['RY', [target], [a / N]])
    const bit = Math.log2(gray(i) ^ gray((i + 1) % N))
    steps.push(['CX', [ctrl[bit], target]])
  }
  return steps
}

/** Prepares sum_i a_i |i> (a normalised to unit length, real amplitudes) from |0...0>: a binary tree of
 *  uniformly controlled RY, the decomposition behind AmplitudeEmbedding. Qubit q is bit q of the index. */
function amplitudeEmbedding(a: number[]): Step[] {
  const n = Math.log2(a.length)
  const norm = Math.hypot(...a)
  const v = a.map((x) => x / norm)
  const steps: Step[] = []
  for (let k = 0; k < n; k++) {
    const size = 1 << (n - k), half = size >> 1
    const theta: number[] = []
    for (let P = 0; P < 1 << k; P++) {
      const blk = v.slice(P * size, (P + 1) * size)
      theta.push(k === n - 1
        ? 2 * Math.atan2(blk[1], blk[0])  // last level carries the signs
        : 2 * Math.atan2(Math.hypot(...blk.slice(half)), Math.hypot(...blk.slice(0, half))))
    }
    const ctrl = [...Array(k).keys()].map((m) => n - k + m)
    steps.push(...ucry(n - 1 - k, ctrl, theta))
  }
  return steps
}

export interface Template {
  key: string
  name: string
  description: string
  make: () => Circuit
}

export const AMPLITUDE_EXAMPLE = A

export const TEMPLATES: Template[] = [
  {
    key: 'empty', name: 'Порожня схема', description: '4 кубіти без вентилів',
    make: () => ({ nQubits: 4, ops: [] }),
  },
  {
    key: 'bell', name: 'Стан Белла', description: 'H + CNOT: заплутана пара, 50 % |00⟩ і 50 % |11⟩',
    make: () => build(2, [['H', [0]], ['CX', [0, 1]], ...measureAll(2)]),
  },
  {
    key: 'ghz', name: 'GHZ, 3 кубіти', description: '(|000⟩ + |111⟩)/√2',
    make: () => build(3, [['H', [0]], ['CX', [0, 1]], ['CX', [1, 2]], ...measureAll(3)]),
  },
  {
    key: 'grover', name: 'Гровер, 2 кубіти', description: 'Пошук |11⟩ за одну ітерацію',
    make: () => build(2, [
      ['H', [0]], ['H', [1]], ['CZ', [0, 1]],
      ['H', [0]], ['H', [1]], ['X', [0]], ['X', [1]], ['CZ', [0, 1]], ['X', [0]], ['X', [1]], ['H', [0]], ['H', [1]],
      ...measureAll(2),
    ]),
  },
  {
    key: 'angle', name: 'Кутове кодування + VQC', description: 'Як у QHybrid-CV: H, RY(xᵢ), 2 шари StronglyEntanglingLayers',
    make: () => build(4, [
      ...X.map((_, q): Step => ['H', [q]]),
      ...X.map((x, q): Step => ['RY', [q], [x]]),
      ['BARRIER', [0]], ['BARRIER', [1]], ['BARRIER', [2]], ['BARRIER', [3]],
      ...sel(4, 2), ...measureAll(4),
    ]),
  },
  {
    key: 'amplitude', name: 'Амплітудне кодування, 3 кубіти',
    description: '8 ознак записано в амплітуди стану: каскад RY і CNOT (розклад AmplitudeEmbedding), далі 2 шари VQC',
    make: () => build(3, [
      ...amplitudeEmbedding(A),
      ['BARRIER', [0]], ['BARRIER', [1]], ['BARRIER', [2]],
      ...sel(3, 2), ...measureAll(3),
    ]),
  },
  {
    key: 'amplitude-only', name: 'Амплітудне кодування без VQC',
    description: 'Лише підготовка стану: амплітуди дорівнюють 8 нормованим ознакам',
    make: () => build(3, amplitudeEmbedding(A)),
  },
  {
    key: 'zz', name: 'ZZ-кодування (Havlíček)', description: 'H, RZ(2xᵢ), CNOT·RZ(2(π−xᵢ)(π−xⱼ))·CNOT на всіх парах',
    make: () => {
      const x = X.map((v) => 0.1 * Math.PI * (1 + v)) // ZZ range [0, γπ] with γ = 0.1
      const steps: Step[] = [...x.map((_, q): Step => ['H', [q]]), ...x.map((v, q): Step => ['RZ', [q], [2 * v]])]
      for (let i = 0; i < 4; i++)
        for (let j = i + 1; j < 4; j++)
          steps.push(['CX', [i, j]], ['RZ', [j], [2 * (Math.PI - x[i]) * (Math.PI - x[j])]], ['CX', [i, j]])
      return build(4, [...steps, ...measureAll(4)])
    },
  },
]
