import { useEffect, useState } from 'react'
import { formatAngle, parseAngle } from '../quantum/angle'
import { GATES, type Op } from '../quantum/gates'

interface Props {
  op: Op | null
  nQubits: number
  onChange: (op: Op) => void
  onDelete: (id: string) => void
}

function roleName(op: Op, i: number): string {
  if (op.kind === 'SWAP' || op.kind === 'RZZ' || op.kind === 'CZ') return `Кубіт ${i + 1}`
  if (GATES[op.kind].arity === 1) return 'Кубіт'
  return i === op.wires.length - 1 ? 'Цільовий' : 'Керуючий'
}

export function Inspector({ op, nQubits, onChange, onDelete }: Props) {
  if (!op) {
    return (
      <section className="panel inspector">
        <p className="hint">Оберіть вентиль на схемі, щоб змінити кубіти чи параметри. Delete видаляє вентиль, Ctrl+Z скасовує дію.</p>
      </section>
    )
  }
  const g = GATES[op.kind]
  return (
    <section className="panel inspector" aria-label="Властивості вентиля">
      <div className="insp-row">
        <strong>{g.label}</strong>
        <span className="muted">{g.title}</span>
        <button className="btn danger" onClick={() => onDelete(op.id)}>Видалити</button>
      </div>
      <div className="insp-row">
        {op.wires.map((w, i) => (
          <label key={i} className="field">
            <span>{roleName(op, i)}</span>
            <select
              value={w}
              onChange={(e) => {
                const wires = [...op.wires]
                const v = Number(e.target.value)
                const other = wires.indexOf(v)
                if (other >= 0) wires[other] = wires[i] // swap roles instead of duplicating a wire
                wires[i] = v
                onChange({ ...op, wires })
              }}
            >
              {[...Array(nQubits).keys()].map((q) => <option key={q} value={q}>q[{q}]</option>)}
            </select>
          </label>
        ))}
        {g.params.map((p, i) => (
          <AngleField key={`${op.id}-${i}`} label={p} value={op.params[i]} onCommit={(v) => {
            const params = [...op.params]
            params[i] = v
            onChange({ ...op, params })
          }} />
        ))}
      </div>
    </section>
  )
}

function AngleField({ label, value, onCommit }: { label: string; value: number; onCommit: (v: number) => void }) {
  const [text, setText] = useState(formatAngle(value).replace('π', 'pi'))
  useEffect(() => setText(formatAngle(value).replace('π', 'pi')), [value])
  const parsed = parseAngle(text)
  return (
    <label className="field">
      <span>{label}</span>
      <input
        className={parsed === null ? 'invalid' : ''}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => parsed !== null && onCommit(parsed)}
        onKeyDown={(e) => e.key === 'Enter' && parsed !== null && onCommit(parsed)}
        aria-invalid={parsed === null}
        spellCheck={false}
      />
      <span className="muted small">{parsed === null ? 'напр. pi/2, -0.5*pi, 1.2' : `= ${parsed.toFixed(4)} рад`}</span>
    </label>
  )
}
