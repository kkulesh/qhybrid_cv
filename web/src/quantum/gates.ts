// Gate catalogue used by the palette, the simulator and the code generators.

export type GateKind =
  | 'H' | 'X' | 'Y' | 'Z' | 'S' | 'SDG' | 'T' | 'TDG' | 'SX'
  | 'RX' | 'RY' | 'RZ' | 'P'
  | 'CX' | 'CZ' | 'SWAP' | 'CCX' | 'RZZ'
  | 'MEASURE' | 'BARRIER'

export type GateGroup = 'clifford' | 'phase' | 'rotation' | 'multi' | 'meta'

export interface GateDef {
  kind: GateKind
  label: string
  title: string // shown as tooltip and in the inspector
  arity: number // number of wires
  params: string[] // parameter names
  group: GateGroup
}

export const GATES: Record<GateKind, GateDef> = {
  H: { kind: 'H', label: 'H', title: 'Адамара (Hadamard)', arity: 1, params: [], group: 'clifford' },
  X: { kind: 'X', label: 'X', title: 'Паулі X (NOT)', arity: 1, params: [], group: 'clifford' },
  Y: { kind: 'Y', label: 'Y', title: 'Паулі Y', arity: 1, params: [], group: 'clifford' },
  Z: { kind: 'Z', label: 'Z', title: 'Паулі Z', arity: 1, params: [], group: 'phase' },
  S: { kind: 'S', label: 'S', title: 'Фазовий S (√Z)', arity: 1, params: [], group: 'phase' },
  SDG: { kind: 'SDG', label: 'S†', title: 'S† (обернений S)', arity: 1, params: [], group: 'phase' },
  T: { kind: 'T', label: 'T', title: 'T (π/8)', arity: 1, params: [], group: 'phase' },
  TDG: { kind: 'TDG', label: 'T†', title: 'T† (обернений T)', arity: 1, params: [], group: 'phase' },
  SX: { kind: 'SX', label: '√X', title: '√X', arity: 1, params: [], group: 'clifford' },
  RX: { kind: 'RX', label: 'RX', title: 'Обертання навколо X', arity: 1, params: ['θ'], group: 'rotation' },
  RY: { kind: 'RY', label: 'RY', title: 'Обертання навколо Y', arity: 1, params: ['θ'], group: 'rotation' },
  RZ: { kind: 'RZ', label: 'RZ', title: 'Обертання навколо Z', arity: 1, params: ['θ'], group: 'rotation' },
  P: { kind: 'P', label: 'P', title: 'Фазовий зсув P(λ)', arity: 1, params: ['λ'], group: 'phase' },
  CX: { kind: 'CX', label: 'CX', title: 'CNOT (керований X)', arity: 2, params: [], group: 'multi' },
  CZ: { kind: 'CZ', label: 'CZ', title: 'Керований Z', arity: 2, params: [], group: 'multi' },
  SWAP: { kind: 'SWAP', label: 'SWAP', title: 'Обмін станами', arity: 2, params: [], group: 'multi' },
  CCX: { kind: 'CCX', label: 'CCX', title: 'Тофолі (двічі керований X)', arity: 3, params: [], group: 'multi' },
  RZZ: { kind: 'RZZ', label: 'RZZ', title: 'ZZ-обертання (IsingZZ)', arity: 2, params: ['θ'], group: 'rotation' },
  MEASURE: { kind: 'MEASURE', label: 'M', title: 'Вимірювання в базисі Z', arity: 1, params: [], group: 'meta' },
  BARRIER: { kind: 'BARRIER', label: '┊', title: 'Бар’єр (візуальний роздільник)', arity: 1, params: [], group: 'meta' },
}

export const PALETTE: GateKind[] = [
  'H', 'X', 'Y', 'Z', 'S', 'SDG', 'T', 'TDG', 'SX', 'P',
  'RX', 'RY', 'RZ', 'RZZ', 'CX', 'CZ', 'SWAP', 'CCX', 'MEASURE', 'BARRIER',
]

/** One placed operation. For controlled gates the last wire is the target, the others are controls. */
export interface Op {
  id: string
  kind: GateKind
  wires: number[]
  params: number[]
  col: number
}

export interface Circuit {
  nQubits: number
  ops: Op[]
}

let counter = 0
export const newId = () => `op${Date.now().toString(36)}${(counter++).toString(36)}`

export function defaultParams(kind: GateKind): number[] {
  return GATES[kind].params.map(() => Math.PI / 2)
}
