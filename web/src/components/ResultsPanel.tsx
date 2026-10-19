import { useMemo, useState } from 'react'
import { bitstring, sampleCounts, type SimResult } from '../quantum/simulator'

type Tab = 'probs' | 'counts' | 'state' | 'bloch'

export function ResultsPanel({ sim, nQubits }: { sim: SimResult; nQubits: number }) {
  const [tab, setTab] = useState<Tab>('probs')
  const [shots, setShots] = useState(1024)
  const [seed, setSeed] = useState(1)
  return (
    <section className="panel results">
      <header className="panel-head tabs" role="tablist">
        {([['probs', 'Ймовірності'], ['counts', 'Вимірювання'], ['state', 'Вектор стану'], ['bloch', 'Сфери Блоха']] as [Tab, string][]).map(([k, name]) => (
          <button key={k} role="tab" aria-selected={tab === k} className={`tab${tab === k ? ' active' : ''}`} onClick={() => setTab(k)}>{name}</button>
        ))}
      </header>
      {sim.warnings.map((w) => <p key={w} className="warn">{w}</p>)}
      {tab === 'probs' && <ProbChart probs={sim.probs} n={nQubits} />}
      {tab === 'counts' && (
        sim.measured.length === 0
          ? <p className="hint">Додайте вимірювання (M) на схему, щоб побачити результати вибірки.</p>
          : (
            <>
              <div className="insp-row">
                <label className="field inline"><span>Кількість вимірювань (shots)</span>
                  <input type="number" min={1} max={100000} value={shots} onChange={(e) => setShots(Math.max(1, Math.min(100000, Number(e.target.value) || 1)))} />
                </label>
                <button className="btn" onClick={() => setSeed((s) => s + 1)}>Повторити запуск</button>
                <span className="muted small">Біти: c[{[...sim.measured].sort((a, b) => b - a).join('], c[')}]</span>
              </div>
              <CountsChart sim={sim} shots={shots} seed={seed} />
            </>
          )
      )}
      {tab === 'state' && <StateTable sim={sim} n={nQubits} />}
      {tab === 'bloch' && (
        <div className="bloch-row">
          {[...Array(nQubits).keys()].map((q) => <Bloch key={q} q={q} v={sim.state.bloch(q)} />)}
        </div>
      )}
    </section>
  )
}

function Bars({ items, yLabel }: { items: { label: string; value: number }[]; yLabel: string }) {
  const W = Math.max(320, items.length * 28 + 56)
  const H = 210
  const plotH = 140
  const max = 1
  return (
    <div className="chart-scroll">
      <svg width={W} height={H} className="chart" role="img" aria-label={yLabel}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid" x1={44} x2={W - 8} y1={10 + plotH * (1 - t)} y2={10 + plotH * (1 - t)} />
            <text className="axis" x={38} y={14 + plotH * (1 - t)} textAnchor="end">{Math.round(t * 100)}</text>
          </g>
        ))}
        <text className="axis" transform={`translate(12 ${10 + plotH / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {items.map((it, i) => {
          const x = 52 + i * 28
          const h = (plotH * it.value) / max
          return (
            <g key={it.label}>
              <rect className="bar" x={x} y={10 + plotH - h} width={18} height={h} rx={2}>
                <title>{`${it.label}: ${(it.value * 100).toFixed(2)} %`}</title>
              </rect>
              <text className="axis mono" transform={`translate(${x + 13} ${plotH + 18}) rotate(-60)`} textAnchor="end">{it.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function ProbChart({ probs, n }: { probs: Float64Array; n: number }) {
  const items = useMemo(() => {
    const all = [...probs].map((p, i) => ({ label: bitstring(i, n), value: p }))
    return all.length <= 32 ? all : all.filter((x) => x.value > 1e-6).sort((a, b) => b.value - a.value).slice(0, 32)
  }, [probs, n])
  return (
    <>
      <Bars items={items} yLabel="Ймовірність, %" />
      <p className="muted small">Точні ймовірності кінцевого стану. Біти записано як q[{n - 1}]…q[0].{probs.length > 32 ? ' Показано 32 найімовірніші стани.' : ''}</p>
    </>
  )
}

function CountsChart({ sim, shots, seed }: { sim: SimResult; shots: number; seed: number }) {
  const items = useMemo(() => {
    const counts = sampleCounts(sim.probs, sim.measured, shots, seed)
    return [...counts.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([label, c]) => ({ label: `${label} (${c})`, value: c / shots }))
  }, [sim, shots, seed])
  return <Bars items={items} yLabel={`Частка з ${shots} вимірювань, %`} />
}

function StateTable({ sim, n }: { sim: SimResult; n: number }) {
  const rows = [...sim.probs.keys()].filter((i) => sim.probs[i] > 1e-9).slice(0, 256)
  return (
    <div className="table-scroll">
      <table className="state">
        <thead><tr><th>Стан</th><th>Амплітуда</th><th>Ймовірність</th><th>Фаза</th></tr></thead>
        <tbody>
          {rows.map((i) => {
            const re = sim.state.re[i], im = sim.state.im[i]
            const ph = Math.atan2(im, re)
            return (
              <tr key={i}>
                <td className="mono">|{bitstring(i, n)}⟩</td>
                <td className="mono">{re.toFixed(4)} {im < 0 ? '−' : '+'} {Math.abs(im).toFixed(4)}i</td>
                <td className="mono">{(sim.probs[i] * 100).toFixed(2)} %</td>
                <td className="mono">{(ph / Math.PI).toFixed(3)}π</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Bloch sphere drawn as an oblique projection: x to the lower left, y to the right, z up. */
function Bloch({ q, v }: { q: number; v: [number, number, number] }) {
  const R = 52
  const [x, y, z] = v
  const px = 70 + R * (y - 0.45 * x)
  const py = 70 - R * (z - 0.3 * x)
  const len = Math.sqrt(x * x + y * y + z * z)
  return (
    <figure className="bloch">
      <svg width={140} height={140} role="img" aria-label={`Сфера Блоха для q[${q}]`}>
        <circle className="sphere" cx={70} cy={70} r={R} />
        <ellipse className="equator" cx={70} cy={70} rx={R} ry={R * 0.32} />
        <line className="axisline" x1={70} x2={70} y1={70 - R} y2={70 + R} />
        <text className="axis" x={70} y={12} textAnchor="middle">|0⟩</text>
        <text className="axis" x={70} y={136} textAnchor="middle">|1⟩</text>
        <line className="vec" x1={70} y1={70} x2={px} y2={py} />
        <circle className="vecdot" cx={px} cy={py} r={4} />
      </svg>
      <figcaption>
        <strong>q[{q}]</strong>
        <span className="mono small">x {x.toFixed(2)} · y {y.toFixed(2)} · z {z.toFixed(2)}</span>
        {len < 0.98 && <span className="muted small">|r| = {len.toFixed(2)}: кубіт заплутаний</span>}
      </figcaption>
    </figure>
  )
}
