import { useRef, useState } from 'react'
import { formatAngle } from '../quantum/angle'
import { GATES, type Circuit, type GateKind, type Op } from '../quantum/gates'
import { depth, span } from '../quantum/layout'

const LEFT = 64
const COL = 56
const ROW = 52
const TOP = 28
const BOX = 40

const cx = (col: number) => LEFT + col * COL + COL / 2
const cy = (q: number) => TOP + q * ROW + ROW / 2

interface Props {
  circuit: Circuit
  selected: string | null
  armed: GateKind | null
  onSelect: (id: string | null) => void
  onDropGate: (kind: GateKind, q: number, col: number) => void
  onMoveOp: (id: string, dq: number, col: number) => void
}

export function CircuitCanvas({ circuit, selected, armed, onSelect, onDropGate, onMoveOp }: Props) {
  const { nQubits, ops } = circuit
  const cols = Math.max(depth(ops) + 4, 14)
  const width = LEFT + cols * COL + 24
  const height = TOP + (nQubits + 1) * ROW + 8
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const drag = useRef<{ id: string; q0: number; moved: boolean } | null>(null)

  const cellAt = (e: { clientX: number; clientY: number }): [number, number] | null => {
    const r = svgRef.current?.getBoundingClientRect()
    if (!r) return null
    const x = e.clientX - r.left
    const y = e.clientY - r.top
    const q = Math.floor((y - TOP) / ROW)
    const c = Math.floor((x - LEFT) / COL)
    if (q < 0 || q >= nQubits || c < 0) return null
    return [q, c]
  }

  return (
    <div className="canvas-scroll">
      <svg
        ref={svgRef}
        className="circuit"
        width={width}
        height={height}
        role="application"
        aria-label="Квантова схема"
        onDragOver={(e) => {
          if (!e.dataTransfer.types.includes('application/x-gate')) return
          e.preventDefault()
          setHover(cellAt(e))
        }}
        onDragLeave={() => setHover(null)}
        onDrop={(e) => {
          e.preventDefault()
          const k = e.dataTransfer.getData('application/x-gate') as GateKind
          const cell = cellAt(e)
          setHover(null)
          if (k && cell) onDropGate(k, cell[0], cell[1])
        }}
        onPointerMove={(e) => {
          if (armed || drag.current) setHover(cellAt(e))
          if (drag.current) drag.current.moved = true
        }}
        onPointerLeave={() => !drag.current && setHover(null)}
        onPointerUp={(e) => {
          const d = drag.current
          drag.current = null
          const cell = cellAt(e)
          if (d && d.moved && cell) onMoveOp(d.id, cell[0] - d.q0, cell[1])
          if (!d) setHover(null)
        }}
        onClick={(e) => {
          if (e.target !== e.currentTarget && !(e.target as Element).classList.contains('cell')) return
          const cell = cellAt(e)
          if (armed && cell) onDropGate(armed, cell[0], cell[1])
          else onSelect(null)
        }}
      >
        {/* wires */}
        {[...Array(nQubits).keys()].map((q) => (
          <g key={`w${q}`}>
            <text className="wire-label" x={LEFT - 12} y={cy(q) + 5} textAnchor="end">q[{q}]</text>
            <line className="wire" x1={LEFT} x2={width - 12} y1={cy(q)} y2={cy(q)} />
          </g>
        ))}
        <text className="wire-label" x={LEFT - 12} y={cy(nQubits) + 5} textAnchor="end">c{nQubits}</text>
        <line className="cwire" x1={LEFT} x2={width - 12} y1={cy(nQubits) - 2} y2={cy(nQubits) - 2} />
        <line className="cwire" x1={LEFT} x2={width - 12} y1={cy(nQubits) + 2} y2={cy(nQubits) + 2} />

        {/* drop cells */}
        {[...Array(nQubits).keys()].flatMap((q) =>
          [...Array(cols).keys()].map((c) => (
            <rect
              key={`c${q}-${c}`}
              className={`cell${hover && hover[0] === q && hover[1] === c ? ' hover' : ''}`}
              x={LEFT + c * COL + 4}
              y={TOP + q * ROW + 4}
              width={COL - 8}
              height={ROW - 8}
              rx={6}
            />
          )),
        )}

        {/* operations */}
        {ops.map((op) => (
          <OpGlyph
            key={op.id}
            op={op}
            nQubits={nQubits}
            selected={op.id === selected}
            onPointerDown={(e, q) => {
              e.stopPropagation()
              onSelect(op.id)
              drag.current = { id: op.id, q0: q, moved: false }
            }}
          />
        ))}
      </svg>
    </div>
  )
}

function OpGlyph({ op, nQubits, selected, onPointerDown }: {
  op: Op
  nQubits: number
  selected: boolean
  onPointerDown: (e: React.PointerEvent, q: number) => void
}) {
  const g = GATES[op.kind]
  const x = cx(op.col)
  const [lo, hi] = span(op)
  const target = op.wires[op.wires.length - 1]
  const controls = op.wires.slice(0, -1)
  const cls = `op g-${g.group}${selected ? ' selected' : ''}`
  const down = (q: number) => (e: React.PointerEvent) => onPointerDown(e, q)
  const param = op.params.length ? formatAngle(op.params[0]) : ''

  if (op.kind === 'MEASURE') {
    const q = op.wires[0]
    return (
      <g className={cls} onPointerDown={down(q)}>
        <line className="mline" x1={x} x2={x} y1={cy(q) + BOX / 2} y2={cy(nQubits) - 8} />
        <path className="marrow" d={`M${x - 5} ${cy(nQubits) - 10} L${x + 5} ${cy(nQubits) - 10} L${x} ${cy(nQubits) - 2} Z`} />
        <text className="mbit" x={x + 8} y={cy(nQubits) - 12}>{q}</text>
        <rect className="box" x={x - BOX / 2} y={cy(q) - BOX / 2} width={BOX} height={BOX} rx={6} />
        <path className="meter" d={`M${x - 11} ${cy(q) + 7} a11 11 0 0 1 22 0`} />
        <path className="meter" d={`M${x} ${cy(q) + 7} L${x + 7} ${cy(q) - 7}`} />
        <title>{g.title}</title>
      </g>
    )
  }
  if (op.kind === 'BARRIER') {
    const q = op.wires[0]
    return (
      <g className={cls} onPointerDown={down(q)}>
        <rect className="hit" x={x - 10} y={cy(q) - ROW / 2} width={20} height={ROW} />
        <line className="barrier" x1={x} x2={x} y1={cy(q) - ROW / 2} y2={cy(q) + ROW / 2} />
        <title>{g.title}</title>
      </g>
    )
  }
  if (g.arity === 1) {
    const q = op.wires[0]
    return (
      <g className={cls} onPointerDown={down(q)}>
        <rect className="box" x={x - BOX / 2} y={cy(q) - BOX / 2} width={BOX} height={BOX} rx={6} />
        <text className="glabel" x={x} y={cy(q) + (param ? 0 : 5)} textAnchor="middle">{g.label}</text>
        {param && <text className="gparam" x={x} y={cy(q) + 13} textAnchor="middle">{param}</text>}
        <title>{`${g.title}${param ? `, ${g.params[0]} = ${param}` : ''}`}</title>
      </g>
    )
  }
  if (op.kind === 'RZZ') {
    return (
      <g className={cls} onPointerDown={down(lo)}>
        <rect className="box" x={x - BOX / 2} y={cy(lo) - BOX / 2} width={BOX} height={cy(hi) - cy(lo) + BOX} rx={6} />
        <text className="glabel" x={x} y={(cy(lo) + cy(hi)) / 2} textAnchor="middle">RZZ</text>
        <text className="gparam" x={x} y={(cy(lo) + cy(hi)) / 2 + 14} textAnchor="middle">{param}</text>
        {op.wires.map((w) => <circle key={w} className="pin" cx={x - BOX / 2} cy={cy(w)} r={3} />)}
        <title>{`${g.title}, θ = ${param}`}</title>
      </g>
    )
  }
  // CX, CZ, SWAP, CCX
  return (
    <g className={cls}>
      <line className="link" x1={x} x2={x} y1={cy(lo)} y2={cy(hi)} />
      {op.kind === 'SWAP'
        ? op.wires.map((w) => (
            <g key={w} onPointerDown={down(w)}>
              <rect className="hit" x={x - 14} y={cy(w) - 14} width={28} height={28} />
              <path className="swapx" d={`M${x - 8} ${cy(w) - 8} L${x + 8} ${cy(w) + 8} M${x + 8} ${cy(w) - 8} L${x - 8} ${cy(w) + 8}`} />
            </g>
          ))
        : (
          <>
            {controls.map((w) => (
              <g key={w} onPointerDown={down(w)}>
                <rect className="hit" x={x - 14} y={cy(w) - 14} width={28} height={28} />
                <circle className="dot" cx={x} cy={cy(w)} r={7} />
              </g>
            ))}
            <g onPointerDown={down(target)}>
              {op.kind === 'CZ'
                ? <circle className="dot" cx={x} cy={cy(target)} r={7} />
                : (
                  <>
                    <circle className="oplus" cx={x} cy={cy(target)} r={15} />
                    <path className="oplus-cross" d={`M${x - 15} ${cy(target)} H${x + 15} M${x} ${cy(target) - 15} V${cy(target) + 15}`} />
                  </>
                )}
              <rect className="hit" x={x - 16} y={cy(target) - 16} width={32} height={32} />
            </g>
          </>
        )}
      <title>{g.title}</title>
    </g>
  )
}
