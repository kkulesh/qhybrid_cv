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

export interface Template {
  key: string
  name: string
  description: string
  make: () => Circuit
}

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
