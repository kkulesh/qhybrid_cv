import { useEffect, useState } from 'react'
import { getModels, type ModelInfo } from '../classify/api'
import { ENCODING_UK, layers, qubits } from '../classify/labels'
import { LineChart } from './LineChart'
import { DatasetPicker, useDatasets } from '../datasets/datasets'
import { ENCODINGS, GAMMAS, hashParams, lastOf, useJob, type Encoding, type TrainParams, type TrainResult } from './api'

const DEFAULTS: TrainParams = { dataset: 'voc', kind: 'hybrid', encoding: 'amplitude', n: 8, gamma: 1, layers: 2, epochs: 15, lr: 0.01, batch_size: 64, per_class: 199, seed: 0 }
const f3 = (v: number) => v.toFixed(3).replace('.', ',')
const fmtG = (g: number) => String(g).replace('.', ',')

/** Parameters from #train?enc=…&n=…&gamma=… (the selection page links here with its recommendation). */
function fromHash(): Partial<TrainParams> | null {
  if (!location.hash.startsWith('#train')) return null
  const q = hashParams(), out: Partial<TrainParams> = {}
  if (q.get('ds')) { out.dataset = q.get('ds')!; out.kind = 'hybrid' }
  const enc = q.get('enc') as Encoding | null
  if (enc && ENCODINGS.includes(enc)) out.encoding = enc
  if (q.get('n')) out.n = +q.get('n')!
  if (q.get('gamma')) out.gamma = +q.get('gamma')!
  return Object.keys(out).length ? out : null
}

/** «Тренування моделі»: train a hybrid (or a classical) model on the chosen dataset and watch loss and validation F1. */
export function TrainPage() {
  const { current: ds, setCurrent, offline, reload } = useDatasets()
  const [p, setP] = useState<TrainParams>({ ...DEFAULTS, ...fromHash() })
  const [fromSel, setFromSel] = useState(!!fromHash())
  const [custom, setCustom] = useState<ModelInfo[]>([])
  const job = useJob<TrainParams>('train')

  const loadModels = (id: string) => getModels(id).then((r) => setCustom(r.models.filter((m) => m.custom).reverse())).catch(() => setCustom([]))
  // the selection page passes its dataset in the address; make it the current one
  useEffect(() => {
    const apply = () => { const h = fromHash(); if (h) { setP((old) => ({ ...old, ...h })); setFromSel(true); if (h.dataset) setCurrent(h.dataset) } }
    apply()
    window.addEventListener('hashchange', apply)
    return () => window.removeEventListener('hashchange', apply)
  }, [setCurrent])
  // a new dataset: its own class count limits the images per class
  useEffect(() => {
    if (!ds || job.status === 'running') return
    setP((old) => ({ ...old, dataset: ds.id, per_class: Math.min(ds.min_per_class ?? 199, 1000) }))
    loadModels(ds.id)
  }, [ds?.id])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (job.params && job.status === 'running') setP(job.params) }, [job.params, job.status])
  useEffect(() => {
    if (job.status === 'done') { if (ds) loadModels(ds.id); window.dispatchEvent(new Event('qhybrid:models-changed')) }
  }, [job.status])

  const set = <K extends keyof TrainParams>(k: K, v: TrainParams[K]) => { setP((old) => ({ ...old, [k]: v })); setFromSel(false) }
  const num = (k: keyof TrainParams, lo: number, hi: number) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = +e.target.value
    if (Number.isFinite(v)) set(k, Math.max(lo, Math.min(hi, v)) as never)
  }

  const epochs = job.events.flatMap((e) => (e.type === 'epoch' ? [e] : []))
  const start = job.events.find((e) => e.type === 'start')
  const done = job.events.find((e) => e.type === 'done')
  const result = done?.type === 'done' ? (done.result as TrainResult).model : null
  const running = job.status === 'running'
  const shown = job.params ?? p
  const maxPer = ds?.min_per_class ?? 199
  const hybrid = p.kind === 'hybrid'
  const shownHybrid = shown.kind !== 'classical'
  const evaluating = running && lastOf(job.events)?.type === 'evaluating'

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
        <header className="panel-head"><h2>Тренування моделі</h2></header>
        <p className="hint lead">Навчається гібридна модель: заморожена YOLO11 дає 1280 ознак, адаптер готує їх для обраного кодування, далі варіаційна квантова схема і лінійна голова на класи набору. Для порівняння можна навчити класичну лінійну модель на тих самих ознаках.</p>
        {fromSel && <p className="note">Параметри заповнено рекомендацією сторінки «Вибір кодування».</p>}
        <DatasetPicker disabled={running} />
        {ds && !ds.available && <p className="warn" role="alert">Немає файлу ознак ({ds.path}). Спершу виконайте python scripts/extract_features.py.</p>}

        <fieldset className="grid2" disabled={running}>
          <label className="field">Тип моделі
            <select value={p.kind} onChange={(e) => set('kind', e.target.value as TrainParams['kind'])}>
              <option value="hybrid">Гібридна (VQC)</option>
              <option value="classical">Класична лінійна</option>
            </select>
          </label>
          {hybrid && <>
          <label className="field">Кодування
            <select value={p.encoding} onChange={(e) => set('encoding', e.target.value as Encoding)}>
              {ENCODINGS.map((e) => <option key={e} value={e}>{ENCODING_UK[e]}</option>)}
            </select>
          </label>
          <label className="field">Кубітів
            <select value={p.n} onChange={(e) => set('n', +e.target.value)}>
              {[4, 6, 8].map((q) => <option key={q} value={q}>{q}</option>)}
            </select>
          </label>
          <label className="field">Масштаб γ
            <select value={p.gamma} onChange={(e) => set('gamma', +e.target.value)}>
              {(GAMMAS.includes(p.gamma) ? GAMMAS : [...GAMMAS, p.gamma].sort((a, b) => a - b)).map((g) => <option key={g} value={g}>{fmtG(g)}</option>)}
            </select>
          </label>
          <label className="field">Варіаційних шарів
            <input type="number" min={1} max={4} value={p.layers} onChange={num('layers', 1, 4)} />
          </label>
          </>}
          <label className="field">Епох
            <input type="number" min={1} max={50} value={p.epochs} onChange={num('epochs', 1, 50)} />
          </label>
          <label className="field">Швидкість навчання
            <input type="number" min={0.0005} max={0.1} step={0.001} value={p.lr} onChange={num('lr', 0.0001, 0.5)} />
          </label>
          <label className="field">Розмір батча
            <select value={p.batch_size} onChange={(e) => set('batch_size', +e.target.value)}>
              {[16, 32, 64, 128, 256].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </label>
          <label className="field">Прикладів на клас
            <input type="number" min={2} max={maxPer} value={Math.min(p.per_class, maxPer)} onChange={num('per_class', 2, maxPer)} />
          </label>
          <label className="field">Зерно (seed)
            <input type="number" min={0} max={9999} value={p.seed} onChange={num('seed', 0, 9999)} />
          </label>
        </fieldset>
        <p className="hint">Навчальна вибірка: до {Math.min(p.per_class, maxPer) * (ds?.classes?.length ?? 0)} зображень, з них 10 % відкладено для валідації. Тестування на {ds?.n_test ?? 0} зображеннях тестової частини.</p>

        <div className="actions">
          {running
            ? <button className="btn danger-btn" onClick={job.cancel}>Скасувати</button>
            : <button className="btn primary" disabled={!ds?.available} onClick={() => ds && job.start({ ...p, dataset: ds.id, per_class: Math.min(p.per_class, maxPer) })}>Почати тренування</button>}
        </div>
        {job.error && <p className="error-note" role="alert">{job.error}</p>}
      </section>

      <section className="panel lab-out">
        <header className="panel-head">
          <h2>Хід навчання</h2>
          {job.status !== 'idle' && <span className="muted small">{shownHybrid ? <>{ENCODING_UK[shown.encoding]}, {qubits(shown.n)}, γ = {fmtG(shown.gamma)}, {layers(shown.layers)}</> : 'класична лінійна модель'}</span>}
        </header>
        {job.status === 'idle' && <p className="hint">Після запуску тут з'являтимуться втрати і F1 на валідації після кожної епохи.</p>}
        {job.status !== 'idle' && (
          <>
            <div className="progress">
              <div className="ptrack" role="progressbar" aria-valuemin={0} aria-valuemax={shown.epochs} aria-valuenow={epochs.length}>
                <span style={{ width: `${(epochs.length / shown.epochs) * 100}%` }} />
              </div>
              <span className="small muted">
                {running && !start && 'Підготовка даних і моделі…'}
                {running && start && !evaluating && `Епоха ${epochs.length} з ${shown.epochs}${epochs.length ? `, минуло ${lastOf(epochs)!.elapsed_s.toFixed(0)} с` : ''}`}
                {evaluating && 'Оцінювання на тестовій вибірці…'}
                {job.status === 'done' && `Готово за ${result?.train_s.toFixed(0)} с`}
                {job.status === 'cancelled' && `Скасовано після епохи ${epochs.length}`}
                {job.status === 'error' && 'Помилка'}
              </span>
            </div>
            {start?.type === 'start' && <p className="hint">Навчання: {start.n_train}, валідація: {start.n_val}, тест: {start.n_test} зображень. Параметрів моделі: {start.params.toLocaleString('uk-UA')}.</p>}
            {epochs.length > 0 && (
              <div className="charts">
                <LineChart title="Втрати (перехресна ентропія) на навчальній вибірці" label="втрати" values={epochs.map((e) => e.loss)} total={shown.epochs} format={(v) => v.toFixed(2).replace('.', ',')} />
                <LineChart title="Macro-F1 на валідаційній вибірці" label="F1" values={epochs.map((e) => e.val_f1)} total={shown.epochs} domain={[0, 1]} format={(v) => v.toFixed(2).replace('.', ',')} />
              </div>
            )}
          </>
        )}
        {result && (
          <div className="reco">
            <h3 className="sub">Результат на тестовій вибірці</h3>
            <dl className="kv">
              <div><dt>Macro-F1</dt><dd><b>{f3(result.test_f1)}</b></dd></div>
              <div><dt>Точність</dt><dd>{f3(result.test_acc)}</dd></div>
              <div><dt>F1 на валідації</dt><dd>{f3(result.val_f1)}</dd></div>
              <div><dt>Параметрів</dt><dd>{result.params.toLocaleString('uk-UA')}</dd></div>
              <div><dt>Час навчання</dt><dd>{result.train_s.toFixed(1).replace('.', ',')} с</dd></div>
              <div><dt>Збережено як</dt><dd className="mono small">{result.id}</dd></div>
            </dl>
            <p className="hint">Модель додано до списку на сторінці «Класифікація», її можна одразу перевірити на своїх зображеннях.</p>
            <div className="actions"><a className="btn" href="#classify">Перейти до класифікації</a></div>
          </div>
        )}
        {epochs.length > 0 && (
          <details className="more">
            <summary>Таблиця за епохами</summary>
            <div className="tbl-wrap">
              <table className="tbl plain">
                <thead><tr><th>Епоха</th><th>Втрати</th><th>F1 (валідація)</th><th>Час, с</th></tr></thead>
                <tbody>{epochs.map((e) => <tr key={e.epoch}><td>{e.epoch}</td><td>{e.loss.toFixed(4).replace('.', ',')}</td><td>{f3(e.val_f1)}</td><td>{e.elapsed_s.toFixed(1).replace('.', ',')}</td></tr>)}</tbody>
              </table>
            </div>
          </details>
        )}
        {custom.length > 0 && (
          <>
            <h3 className="sub">Навчені на цій сторінці моделі набору «{ds?.name}»</h3>
            <div className="tbl-wrap">
              <table className="tbl plain">
                <thead><tr><th>Модель</th><th>γ</th><th>Шари</th><th>Епох</th><th>F1 (тест)</th><th>Точність</th></tr></thead>
                <tbody>{custom.map((m) => {
                  const x = m as ModelInfo & { epochs?: number }
                  return <tr key={m.id} title={m.id}><td>{m.kind === 'classical' ? 'Класична лінійна' : `${ENCODING_UK[m.encoding ?? '']}, ${qubits(m.n ?? 0)}`}</td><td>{m.gamma != null ? fmtG(m.gamma) : '—'}</td><td>{m.kind === 'classical' ? '—' : m.layers}</td><td>{x.epochs ?? '—'}</td><td>{f3(m.test_f1)}</td><td>{f3(m.test_acc)}</td></tr>
                })}</tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
