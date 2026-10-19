// Left alignment: every operation moves to the earliest column after the previous operations that
// touch any wire between its lowest and highest qubit (multi-qubit gates block the wires they span).
import type { Op } from './gates'

export function span(op: Op): [number, number] {
  return [Math.min(...op.wires), Math.max(...op.wires)]
}

/** Wires an operation occupies in the drawing; a measurement also blocks the wires below it,
 *  because its arrow runs down to the classical register. */
export function blockSpan(op: Op, nQubits: number): [number, number] {
  const [lo, hi] = span(op)
  return op.kind === 'MEASURE' ? [lo, nQubits - 1] : [lo, hi]
}

export function compact(ops: Op[], nQubits: number): Op[] {
  const sorted = [...ops].sort((a, b) => a.col - b.col || span(a)[0] - span(b)[0])
  const next = new Array(nQubits).fill(0)
  return sorted.map((op) => {
    const [lo, hi] = blockSpan(op, nQubits)
    let col = 0
    for (let q = lo; q <= hi; q++) col = Math.max(col, next[q])
    for (let q = lo; q <= hi; q++) next[q] = col + 1
    return { ...op, col }
  })
}

export function depth(ops: Op[]): number {
  return ops.reduce((d, op) => Math.max(d, op.col + 1), 0)
}
