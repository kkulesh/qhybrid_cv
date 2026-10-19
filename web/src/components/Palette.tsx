import { GATES, PALETTE, type GateKind } from '../quantum/gates'

interface Props {
  armed: GateKind | null
  onArm: (k: GateKind | null) => void
}

/** Gate palette: drag a gate onto the canvas, or click it and then click a cell. */
export function Palette({ armed, onArm }: Props) {
  return (
    <section className="panel palette" aria-label="Операції">
      <header className="panel-head">
        <h2>Операції</h2>
      </header>
      <div className="palette-grid">
        {PALETTE.map((k) => {
          const g = GATES[k]
          return (
            <button
              key={k}
              className={`gate-chip g-${g.group}${g.label.length > 3 ? ' long' : ''}${armed === k ? ' armed' : ''}`}
              title={`${g.title}${g.params.length ? ` (${g.params.join(', ')})` : ''}`}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/x-gate', k)
                e.dataTransfer.effectAllowed = 'copy'
              }}
              onClick={() => onArm(armed === k ? null : k)}
            >
              {k === 'MEASURE' ? <MeterIcon /> : g.label}
            </button>
          )
        })}
      </div>
      <p className="hint">Перетягніть вентиль на схему або натисніть на нього, а потім на клітинку.</p>
    </section>
  )
}

export function MeterIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 17a8 8 0 0 1 16 0" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M12 17 L17 8" stroke="currentColor" strokeWidth="2" />
    </svg>
  )
}
