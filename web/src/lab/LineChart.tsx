import { useState } from 'react'

interface Props {
  title: string
  values: number[]          // one value per epoch, epoch 1 first
  total: number             // planned number of epochs, fixes the x axis while training runs
  format: (v: number) => string
  domain?: [number, number] // fixed y range; otherwise fitted to the data
  label: string             // what a value is, for the tooltip and screen readers
}

const W = 520, H = 200, L = 44, R = 12, T = 10, B = 26

function niceTicks(lo: number, hi: number, count = 4) {
  const step0 = (hi - lo) / count
  const mag = 10 ** Math.floor(Math.log10(step0 || 1))
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= step0) ?? step0
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

/** Single-series line chart of a value per epoch, with a hover crosshair and tooltip. */
export function LineChart({ title, values, total, format, domain, label }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const n = Math.max(total, values.length, 2)
  let [lo, hi] = domain ?? [Math.min(...values), Math.max(...values)]
  if (!domain) { const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.1 || 0.1; lo -= pad; hi += pad }
  const ticks = niceTicks(lo, hi)
  if (!domain) { lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]) }
  const x = (i: number) => L + (i / (n - 1)) * (W - L - R)
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B)
  const path = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  const xticks = niceTicks(1, n, Math.min(n - 1, 6)).filter((e) => Number.isInteger(e) && e >= 1 && e <= n)

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * (W - L - R)
    const i = Math.round((px / (W - L - R)) * (n - 1))
    setHover(i >= 0 && i < values.length ? i : null)
  }

  return (
    <figure className="lchart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}: ${values.map((v, i) => `епоха ${i + 1} ${format(v)}`).join(', ')}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={L} x2={W - R} y1={y(t)} y2={y(t)} />
            <text className="axis" x={L - 6} y={y(t) + 4} textAnchor="end">{format(t)}</text>
          </g>
        ))}
        {xticks.map((e) => <text key={e} className="axis" x={x(e - 1)} y={H - 8} textAnchor="middle">{e}</text>)}
        {values.length > 1 && <path className="line" d={path} />}
        {values.map((v, i) => <circle key={i} className={`pt${i === values.length - 1 ? ' last' : ''}`} cx={x(i)} cy={y(v)} r={i === hover ? 5 : 3} />)}
        {hover !== null && <line className="cross" x1={x(hover)} x2={x(hover)} y1={T} y2={H - B} />}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      <div className="lchart-foot">
        <span className="muted small">Епоха</span>
        {hover !== null
          ? <span className="tip">Епоха {hover + 1}: {label} {format(values[hover])}</span>
          : values.length > 0 && <span className="muted small">Остання: {format(values[values.length - 1])}</span>}
      </div>
    </figure>
  )
}
