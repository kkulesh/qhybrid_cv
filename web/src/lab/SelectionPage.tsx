import { useEffect, useMemo, useState } from 'react'
import { ENCODING_UK, plural, qubits } from '../classify/labels'
import { DatasetPicker, useDatasets } from '../datasets/datasets'
import { ENCODINGS, GAMMAS, lastOf, useJob, type Encoding, type SelectionParams, type SelResult, type SelRow } from './api'

const fmtG = (g: number) => String(g).replace('.', ',')
const conf = (n: number) => plural(n, ['конфігурація', 'конфігурації', 'конфігурацій'])
const sec = (v: number) => v.toFixed(1).replace('.', ',') + ' с'
const fmtK = (k: number) => k.toFixed(3).replace('.', ',')

/** «Автоматичний вибір кодування»: KTA of the fidelity kernel for every encoding and scale, no VQC training. */
export function SelectionPage() {
  const { datasets, current: ds, offline, reload } = useDatasets()
  const [n, setN] = useState(8)
  const [perClass, setPerClass] = useState(20)
  const [gammas, setGammas] = useState<number[]>(GAMMAS)
  const [encs, setEncs] = useState<Encoding[]>(ENCODINGS)
  const job = useJob<SelectionParams>('selection')

  // after a reload, show the form with the parameters of the job being displayed
  useEffect(() => {
    const p = job.params
    if (p && job.status === 'running') { setN(p.n); setPerClass(p.per_class); setGammas(p.gammas); setEncs(p.encodings) }
  }, [job.params, job.status])

  const rows = job.events.flatMap((e) => (e.type === 'row' ? [e.row] : []))
  const last = lastOf(job.events.filter((e) => e.type === 'row'))
  const done = job.events.find((e) => e.type === 'done')
  const result = done?.type === 'done' ? (done.result as SelResult) : null
  const running = job.status === 'running'
  const shownGammas = job.params?.gammas ?? gammas
  const shownEncs = job.params?.encodings ?? encs
  const shownN = job.params?.n ?? n

  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v])
  const total = encs.length * gammas.length
  const canStart = !running && encs.length > 0 && gammas.length > 0 && ds?.available
  const maxPer = Math.min(100, ds?.min_per_class ?? 100)
  const shownDs = datasets.find((d) => d.id === (job.params?.dataset ?? ds?.id))

  if (offline) return (
    <div className="lab">
      <section className="panel offline" role="alert">
        <h2>Сервер недоступний</h2>
        <p className="muted">Запустіть бекенд: uvicorn server.app:app --port 8000</p>
        <button className="btn" onClick={() => reload()}>Спробувати ще раз</button>
      </section>
    </div>
  )

  return (
    <div className="lab">
      <section className="panel lab-form">
        <header className="panel-head"><h2>Автоматичний вибір кодування</h2></header>
        <p className="hint lead">Для кожної схеми кодування і масштабу γ обчислюється квантове ядро на малій вибірці ознак YOLO11 і його узгодженість з мітками класів (центрована KTA). Квантова схема при цьому не навчається, тому перебір триває секунди, а не години.</p>
        <DatasetPicker disabled={running} />
        {ds && !ds.available && <p className="warn" role="alert">Немає файлу ознак ({ds.path}). Спершу виконайте python scripts/extract_features.py.</p>}

        <label className="field">Кількість кубітів
          <select value={n} disabled={running} onChange={(e) => setN(+e.target.value)}>
            {[4, 6, 8].map((q) => <option key={q} value={q}>{q}</option>)}
          </select>
        </label>
        <label className="field">Прикладів на клас у вибірці
          <input type="number" min={2} max={maxPer} value={Math.min(perClass, maxPer)} disabled={running}
            onChange={(e) => setPerClass(Math.max(2, Math.min(maxPer, +e.target.value || 2)))} />
          <span className="small muted">{Math.min(perClass, maxPer) * (ds?.classes?.length ?? 0)} зображень з навчальної частини набору</span>
        </label>

        <fieldset className="checks" disabled={running}>
          <legend>Схеми кодування</legend>
          {ENCODINGS.map((e) => (
            <label key={e}><input type="checkbox" checked={encs.includes(e)} onChange={() => setEncs(toggle(encs, e))} />{ENCODING_UK[e]}</label>
          ))}
        </fieldset>
        <fieldset className="checks" disabled={running}>
          <legend>Масштаб γ</legend>
          {GAMMAS.map((g) => (
            <label key={g}><input type="checkbox" checked={gammas.includes(g)} onChange={() => setGammas(toggle(gammas, g).sort((a, b) => a - b))} />{fmtG(g)}</label>
          ))}
        </fieldset>

        <div className="actions">
          {running
            ? <button className="btn danger-btn" onClick={job.cancel}>Скасувати</button>
            : <button className="btn primary" disabled={!canStart} onClick={() => ds && job.start({ dataset: ds.id, n, per_class: Math.min(perClass, maxPer), gammas, encodings: encs })}>Обрати кодування ({total} {conf(total)})</button>}
        </div>
        {job.error && <p className="error-note" role="alert">{job.error}</p>}
      </section>

      <section className="panel lab-out">
        <header className="panel-head">
          <h2>Узгодженість ядра з мітками (KTA), {qubits(shownN)}</h2>
          {shownDs && <span className="muted small">набір «{shownDs.name}»</span>}
        </header>
        {job.status === 'idle' && <p className="hint">Оберіть параметри і натисніть «Обрати кодування». Тут з'являтиметься значення KTA для кожної конфігурації одразу після обчислення.</p>}
        {job.status !== 'idle' && (
          <>
            <Progress done={last?.type === 'row' ? last.done : 0} total={last?.type === 'row' ? last.total : shownEncs.length * shownGammas.length}
              status={job.status} seconds={lastOf(job.events)?.t ?? 0} />
            <Heatmap rows={rows} encs={shownEncs} gammas={shownGammas} best={result?.best} />
          </>
        )}
        {result && <Recommendation r={result} n={shownN} dataset={job.params?.dataset ?? 'voc'} />}
        {rows.length > 0 && (
          <details className="more">
            <summary>Усі показники ({rows.length})</summary>
            <div className="tbl-wrap">
              <table className="tbl plain">
                <thead><tr><th>Кодування</th><th>γ</th><th>KTA</th><th title="Дисперсія позадіагональних елементів ядра: мала дисперсія означає концентрацію ядра">Дисперсія ядра</th><th title="Відстань Гільберта–Шмідта між середніми станами класів">HS-відстань</th><th>Час, с</th></tr></thead>
                <tbody>{rows.map((r) => (
                  <tr key={r.encoding + r.gamma}><td>{ENCODING_UK[r.encoding]}</td><td>{fmtG(r.gamma)}</td><td>{fmtK(r.kta)}</td><td>{r.var.toExponential(2)}</td><td>{r.hs.toFixed(3).replace('.', ',')}</td><td>{r.time_s.toFixed(2).replace('.', ',')}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </details>
        )}
      </section>
    </div>
  )
}

function Progress({ done, total, status, seconds }: { done: number; total: number; status: string; seconds: number }) {
  const text = status === 'running' ? `Обчислено ${done} з ${total}` :
    status === 'done' ? `Готово: ${total} ${conf(total)} за ${sec(seconds)}` :
    status === 'cancelled' ? `Скасовано після ${done} з ${total}` : 'Помилка'
  return (
    <div className="progress">
      <div className="ptrack" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}><span style={{ width: `${(done / (total || 1)) * 100}%` }} /></div>
      <span className="small muted">{text}</span>
    </div>
  )
}

/** Encodings × γ grid. Colour is one hue from light (low KTA) to dark (high KTA), scaled to the computed range. */
function Heatmap({ rows, encs, gammas, best }: { rows: SelRow[]; encs: Encoding[]; gammas: number[]; best?: SelRow }) {
  const by = useMemo(() => new Map(rows.map((r) => [`${r.encoding}|${r.gamma}`, r])), [rows])
  const ktas = rows.map((r) => r.kta)
  const lo = Math.min(...ktas), hi = Math.max(...ktas)
  const share = (k: number) => (hi > lo ? (k - lo) / (hi - lo) : 1)
  return (
    <div className="tbl-wrap">
      <table className="heat">
        <thead><tr><th>Кодування \ γ</th>{gammas.map((g) => <th key={g}>{fmtG(g)}</th>)}</tr></thead>
        <tbody>{encs.map((e) => (
          <tr key={e}>
            <th scope="row">{ENCODING_UK[e]}</th>
            {gammas.map((g) => {
              const r = by.get(`${e}|${g}`)
              if (!r) return <td key={g} className="empty">…</td>
              const s = share(r.kta)
              const isBest = best && best.encoding === e && best.gamma === g
              return (
                <td key={g} className={`${s > 0.55 ? 'dark' : ''}${isBest ? ' best' : ''}`}
                  style={{ background: `color-mix(in oklab, var(--accent) ${Math.round(8 + s * 80)}%, var(--panel))` }}
                  title={`${ENCODING_UK[e]}, γ = ${fmtG(g)}: KTA ${fmtK(r.kta)}, дисперсія ядра ${r.var.toExponential(2)}, ${r.time_s.toFixed(2).replace('.', ',')} с`}>
                  {fmtK(r.kta)}
                </td>
              )
            })}
          </tr>
        ))}</tbody>
      </table>
      {rows.length > 1 && (
        <div className="scale small muted" aria-hidden>
          <span>{fmtK(lo)}</span><i /><span>{fmtK(hi)}</span><span className="spacer" /><span>Насиченіший колір означає вищу KTA; рамкою виділено рекомендовану конфігурацію.</span>
        </div>
      )}
    </div>
  )
}

function Recommendation({ r, n, dataset }: { r: SelResult; n: number; dataset: string }) {
  const b = r.best
  const trainHref = `#train?ds=${encodeURIComponent(dataset)}&enc=${b.encoding}&n=${n}&gamma=${b.gamma}`
  return (
    <div className="reco">
      <h3 className="sub">Рекомендація</h3>
      <p className="reco-main"><b>{ENCODING_UK[b.encoding]} кодування, γ = {fmtG(b.gamma)}</b> (KTA {fmtK(b.kta)})</p>
      <ol className="rank">
        {r.best_per_encoding.map((x) => (
          <li key={x.encoding}>{ENCODING_UK[x.encoding]}: найкраще γ = {fmtG(x.gamma)}, KTA {fmtK(x.kta)}</li>
        ))}
      </ol>
      <p className="hint">Вибірка {r.sample_size} зображень, загальний час {sec(r.time_s)}. У дослідженні на CIFAR-10, Fashion-MNIST і PASCAL VOC кодування з найвищою KTA давало найкращу модель у 9 з 9 випадків.</p>
      <div className="actions"><a className="btn primary" href={trainHref}>Навчити рекомендовану модель</a></div>
    </div>
  )
}
