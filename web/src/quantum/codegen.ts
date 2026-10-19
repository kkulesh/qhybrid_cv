// Export a circuit as OpenQASM 3.0 or as a PennyLane QNode.
import { formatAngle } from './angle'
import type { Circuit } from './gates'

const num = (v: number) => {
  const s = formatAngle(v, 8) // full precision in exported code
  return s.includes('π') ? s.replace('π', 'pi') : s
}

const QASM_NAME: Record<string, string> = {
  H: 'h', X: 'x', Y: 'y', Z: 'z', S: 's', SDG: 'sdg', T: 't', TDG: 'tdg', SX: 'sx',
  RX: 'rx', RY: 'ry', RZ: 'rz', P: 'p', CX: 'cx', CZ: 'cz', SWAP: 'swap', CCX: 'ccx', RZZ: 'rzz',
}

export function toQasm(c: Circuit): string {
  const lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', '']
  if (c.ops.some((o) => o.kind === 'RZZ')) {
    // rzz is not part of stdgates.inc
    lines.push('gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }', '')
  }
  lines.push(`qubit[${c.nQubits}] q;`, `bit[${c.nQubits}] c;`, '')
  for (const op of [...c.ops].sort((a, b) => a.col - b.col)) {
    const qs = op.wires.map((w) => `q[${w}]`).join(', ')
    if (op.kind === 'MEASURE') lines.push(`c[${op.wires[0]}] = measure q[${op.wires[0]}];`)
    else if (op.kind === 'BARRIER') lines.push(`barrier ${qs};`)
    else {
      const p = op.params.length ? `(${op.params.map(num).join(', ')})` : ''
      lines.push(`${QASM_NAME[op.kind]}${p} ${qs};`)
    }
  }
  return lines.join('\n') + '\n'
}

const PL: Record<string, (w: string, p: string[]) => string> = {
  H: (w) => `qml.Hadamard(wires=${w})`,
  X: (w) => `qml.PauliX(wires=${w})`,
  Y: (w) => `qml.PauliY(wires=${w})`,
  Z: (w) => `qml.PauliZ(wires=${w})`,
  S: (w) => `qml.S(wires=${w})`,
  SDG: (w) => `qml.adjoint(qml.S)(wires=${w})`,
  T: (w) => `qml.T(wires=${w})`,
  TDG: (w) => `qml.adjoint(qml.T)(wires=${w})`,
  SX: (w) => `qml.SX(wires=${w})`,
  RX: (w, p) => `qml.RX(${p[0]}, wires=${w})`,
  RY: (w, p) => `qml.RY(${p[0]}, wires=${w})`,
  RZ: (w, p) => `qml.RZ(${p[0]}, wires=${w})`,
  P: (w, p) => `qml.PhaseShift(${p[0]}, wires=${w})`,
  CX: (w) => `qml.CNOT(wires=${w})`,
  CZ: (w) => `qml.CZ(wires=${w})`,
  SWAP: (w) => `qml.SWAP(wires=${w})`,
  CCX: (w) => `qml.Toffoli(wires=${w})`,
  RZZ: (w, p) => `qml.IsingZZ(${p[0]}, wires=${w})`,
}

export function toPennyLane(c: Circuit): string {
  const pyNum = (v: number) => num(v).replace(/pi/g, 'np.pi')
  const body: string[] = []
  const measured: number[] = []
  for (const op of [...c.ops].sort((a, b) => a.col - b.col)) {
    if (op.kind === 'MEASURE') { measured.push(op.wires[0]); continue }
    if (op.kind === 'BARRIER') { body.push(`    qml.Barrier(wires=[${op.wires.join(', ')}])`); continue }
    const w = op.wires.length === 1 ? `${op.wires[0]}` : `[${op.wires.join(', ')}]`
    body.push(`    ${PL[op.kind](w, op.params.map(pyNum))}`)
  }
  const wires = measured.length ? measured : [...Array(c.nQubits).keys()]
  return [
    'import numpy as np',
    'import pennylane as qml',
    '',
    `dev = qml.device("default.qubit", wires=${c.nQubits})`,
    '',
    '',
    '@qml.qnode(dev)',
    'def circuit():',
    ...(body.length ? body : ['    pass']),
    `    return qml.probs(wires=[${wires.join(', ')}])`,
    '',
    '',
    '# PennyLane orders wires as the most significant bit first: probs[i] <-> wires ' +
      `[${wires.join(', ')}] read left to right`,
    'print(circuit())',
    'print(qml.draw(circuit)())',
    '',
  ].join('\n')
}
