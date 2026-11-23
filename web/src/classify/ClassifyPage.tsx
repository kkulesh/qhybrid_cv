import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { classify, compareAll, getDevices, getModels, getSamples, sampleUrl, EXACT, type Comparison, type Device, type ModelInfo, type Prediction, type Registry, type RunOn, type Sample, type Timings } from './api'
import { DatasetPicker, useDatasets } from '../datasets/datasets'
import { DEVICE_UK, ENCODING_UK, modelName, uk } from './labels'

const STAGES: { key: keyof Timings; name: string; cls: string }[] = [
  { key: 'preprocess', name: 'Підготовка', cls: 's-pre' },
  { key: 'backbone', name: 'YOLO11', cls: 's-bb' },
  { key: 'adapter', name: 'Адаптер', cls: 's-ad' },
  { key: 'circuit', name: 'VQC (симуляція)', cls: 's-vqc' },
  { key: 'head', name: 'Голова', cls: 's-head' },
]

const fmtMs = (v: number) => (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : v.toFixed(0)) + ' мс'
const pct = (p: number) => (p * 100).toFixed(p >= 0.995 || p < 0.1 ? 1 : 0) + ' %'

interface Image { blob: Blob; url: string; name: string; truth?: string; own?: boolean }

/** Downscale big photos in the browser so the upload stays fast; the backbone resizes to 224 px anyway. */
async function shrink(file: Blob, maxSide = 800): Promise<Blob> {
  try {
    const bmp = await createImageBitmap(file)
    const k = maxSide / Math.max(bmp.width, bmp.height)
    if (k >= 1) { bmp.close(); return file }
    const c = document.createElement('canvas')
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k)
    c.getContext('2d')!.drawImage(bmp, 0, 0, c.width, c.height)
    bmp.close()
    return await new Promise<Blob>((res) => c.toBlob((b) => res(b ?? file), 'image/jpeg', 0.9))
  } catch { return file }  // the server reports formats it cannot read
}

export function ClassifyPage() {
  const { current: ds } = useDatasets()
  const dsId = ds?.id ?? 'voc'
  const [registry, setRegistry] = useState<Registry | null>(null)
  // samples are tagged with their dataset, so a switch never shows (or requests) the previous dataset's images
  const [sampleSet, setSampleSet] = useState<{ ds: string, list: Sample[] }>({ ds: '', list: [] })
  const samples = sampleSet.ds === dsId ? sampleSet.list : []
  const dsRef = useRef(dsId)  // a late answer for a dataset no longer chosen is dropped
  dsRef.current = dsId
  const [offline, setOffline] = useState<string | null>(null)
  const [modelId, setModelId] = useState('amplitude-8q')
  const [image, setImage] = useState<Image | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pred, setPred] = useState<(Prediction & { at?: Date }) | null>(null)
  const [cmp, setCmp] = useState<Comparison | null>(null)
  const [drag, setDrag] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const reqId = useRef(0)
  // where the circuit of a hybrid model runs: default.qubit (exact) or the local Amazon Braket simulator
  const [devices, setDevices] = useState<Device[]>([])
  const [runOn, setRunOn] = useState<RunOn>(EXACT)
  useEffect(() => { getDevices().then(setDevices).catch(() => setDevices([])) }, [])

  const applyRegistry = useCallback((r: Registry) => {
    setRegistry(r)
    // keep the chosen model if this dataset has it, otherwise the best default
    setModelId((cur) => r.models.some((m) => m.id === cur) ? cur
      : (r.models.find((m) => m.id === 'amplitude-8q') ?? r.models.find((m) => m.kind === 'hybrid') ?? r.models[0])?.id ?? '')
  }, [])
  const load = useCallback(() => {
    setOffline(null)
    Promise.all([getModels(dsId), getSamples(dsId)])
      .then(([r, s]) => { if (dsRef.current !== dsId) return; applyRegistry(r); setSampleSet({ ds: dsId, list: s }) })
      .catch(() => setOffline('Сервер класифікації недоступний. Запустіть бекенд: uvicorn server.app:app --port 8000'))
  }, [dsId, applyRegistry])
  useEffect(load, [load])
  // another dataset: its own classes and models, so the old result no longer applies
  useEffect(() => { setPred(null); setCmp(null); setError(null); setImage((old) => (old?.truth ? null : old)) }, [dsId])
  // a model trained on the «Тренування» page appears in the list without reloading the page
  useEffect(() => {
    const on = () => { getModels(dsId).then(applyRegistry).catch(() => {}) }
    window.addEventListener('qhybrid:models-changed', on)
    return () => window.removeEventListener('qhybrid:models-changed', on)
  }, [dsId, applyRegistry])

  const groups = useMemo(() => {
    const g = new Map<string, ModelInfo[]>()
    for (const m of registry?.models ?? []) {
      const k = m.custom ? 'Навчені на сторінці «Тренування»' : m.kind === 'classical' ? 'Класична' : `Гібридні: ${ENCODING_UK[m.encoding ?? ''] ?? m.encoding}`
      g.set(k, [...(g.get(k) ?? []), m])
    }
    return [...g]
  }, [registry])
  const model = registry?.models.find((m) => m.id === modelId)
  // digit datasets (classes 0-9): own photos are made to look like MNIST (white digit on black, centred)
  const isDigits = !!registry && registry.classes.length > 1 && registry.classes.every((c) => /^\d$/.test(c))
  const [digitPrep, setDigitPrep] = useState(true)
  const prep = isDigits && digitPrep && !image?.truth ? 'digit' : ''
  const prepared = (pred ?? cmp)?.prepared

  const setNewImage = (blob: Blob, name: string, truth?: string, own = false) => {
    setImage((old) => { if (old) URL.revokeObjectURL(old.url); return { blob, url: URL.createObjectURL(blob), name, truth, own } })
    setPred(null); setCmp(null); setError(null)
  }
  const pickSample = async (s: Sample) => {
    try {
      const r = await fetch(sampleUrl(s.name, dsId))
      if (!r.ok) throw new Error()
      setNewImage(await r.blob(), s.name, s.label)
    } catch { setError('Не вдалося завантажити приклад.') }
  }
  const onFile = async (f: File | undefined) => {
    if (!f) return
    if (f.type && !f.type.startsWith('image/')) { setError('Оберіть файл зображення (JPEG, PNG, WebP, BMP).'); return }
    setNewImage(await shrink(f), f.name || 'зображення з буфера', undefined, true)
  }

  // Ctrl+V pastes an image from the clipboard while this page is visible.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const root = document.querySelector('.cls')
      if (!root || !(root as HTMLElement).offsetParent) return
      const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith('image/'))
      if (f) { e.preventDefault(); onFile(f) }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  const run = async (mode: 'one' | 'all') => {
    if (!image || !modelId) return
    const id = ++reqId.current
    setBusy(true); setError(null)
    try {
      if (mode === 'one') { const r = await classify(image.blob, modelId, 5, dsId, prep, runOn); if (id === reqId.current) { setPred({ ...r, at: new Date() }); setCmp(null) } }
      else { const r = await compareAll(image.blob, 3, dsId, prep, runOn); if (id === reqId.current) { setCmp(r); setPred(null) } }
    } catch (e) { if (id === reqId.current) setError(e instanceof Error && e.message !== 'Failed to fetch' ? e.message : 'Сервер не відповідає.') }
    finally { if (id === reqId.current) setBusy(false) }
  }

  // Inference runs on the server as soon as an image or a model is chosen.
  useEffect(() => { if (image) run('one') }, [image, modelId, prep, runOn])  // eslint-disable-line react-hooks/exhaustive-deps

  if (offline) return (
    <div className="cls">
      <section className="panel offline" role="alert">
        <h2>Немає зв'язку з сервером</h2>
        <p>{offline}</p>
        <button className="btn" onClick={load}>Спробувати ще раз</button>
      </section>
    </div>
  )

  return (
    <div className="cls">
      <section className="panel cls-input">
        <header className="panel-head">
          <h2>Зображення</h2>
          <span className="spacer" />
          <button className="btn primary" onClick={() => fileRef.current?.click()}>Завантажити своє зображення</button>
        </header>
        <div
          className={`drop${drag ? ' over' : ''}${image ? ' has' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]) }}
          role="button" tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileRef.current?.click() }}
        >
          {image
            ? <img src={image.url} alt={image.name} />
            : <span>Перетягніть сюди фото, натисніть, щоб обрати файл, або вставте його через Ctrl+V.<br />Найкраще, коли об'єкт займає більшу частину кадру: моделі навчено на вирізках об'єктів.</span>}
        </div>
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        {isDigits && (
          <label className="prep-opt">
            <input type="checkbox" checked={digitPrep} onChange={(e) => setDigitPrep(e.target.checked)} />
            <span>Підготувати своє фото як цифру MNIST: світлий фон стає чорним, цифра вирізається, центрується і зменшується до 28×28</span>
          </label>
        )}
        {prepared && <div className="prep-view"><img src={prepared} alt="Зображення після підготовки" /><span className="hint">Так зображення бачить модель після підготовки.</span></div>}
        {image && <p className="hint">{image.truth ? <>Приклад із тестової частини набору, справжній клас: <b>{uk(image.truth)}</b>{uk(image.truth) !== image.truth && <> ({image.truth})</>}</> : <>Ваше зображення: {image.name}</>}</p>}

        {samples.length > 0 && <h3 className="sub">Приклади з набору «{ds?.name ?? 'PASCAL VOC 2007'}»</h3>}
        <div className="gallery">
          {samples.map((s) => (
            <button key={s.name} className={`thumb${image?.name === s.name ? ' on' : ''}`} onClick={() => pickSample(s)} title={uk(s.label) !== s.label ? `${uk(s.label)} (${s.label})` : s.label}>
              <img src={sampleUrl(s.name, dsId)} alt={uk(s.label)} loading="lazy" />
            </button>
          ))}
        </div>
      </section>

      <section className="panel cls-ctrl">
        <header className="panel-head"><h2>Модель</h2></header>
        <DatasetPicker />
        {registry && registry.models.length === 0 && <p className="warn">Для цього набору ще немає навчених моделей. Навчіть модель на сторінці <a href="#train">«Тренування»</a>.</p>}
        <label className="field">
          <span>Кодування і кількість кубітів</span>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {groups.map(([g, ms]) => (
              <optgroup key={g} label={g}>
                {ms.map((m) => <option key={m.id} value={m.id}>{modelName(m)} · seed {m.seed} · F1 {m.test_f1.toFixed(3)}</option>)}
              </optgroup>
            ))}
          </select>
        </label>
        {model && <ModelCard m={model} />}
        <DevicePicker devices={devices} value={runOn} onChange={setRunOn} classical={model?.kind === 'classical'} />
        <div className="actions">
          <button className="btn" disabled={!image || busy || !modelId} onClick={() => run('one')}>Класифікувати ще раз</button>
          <button className="btn" disabled={!image || busy || !modelId} onClick={() => run('all')}>Порівняти всі моделі</button>
        </div>
        {busy && <p className="hint">Обчислення…</p>}
        {error && <p className="error-note" role="alert">{error}</p>}
        {!image && <p className="hint">Завантажте своє зображення або оберіть приклад. Класифікація запуститься одразу, а при зміні моделі повториться автоматично.</p>}
      </section>

      <section className="panel cls-out">
        <header className="panel-head"><h2>Результат</h2></header>
        {pred && <Result p={pred} truth={image?.truth} />}
        {cmp && <CompareTable c={cmp} truth={image?.truth} onPick={(id) => { setModelId(id) }} />}
        {!pred && !cmp && !busy && <p className="hint">Тут з'явиться клас об'єкта, ймовірності та час кожного етапу.</p>}
      </section>
    </div>
  )
}

function ModelCard({ m }: { m: ModelInfo }) {
  const rows: [string, string][] = m.kind === 'classical'
    ? [['Тип', 'логістична регресія на 1280 ознаках YOLO11']]
    : [
        ['Схема', `YOLO11 → адаптер → кодування (${(ENCODING_UK[m.encoding ?? ''] ?? '').toLowerCase()}) → VQC, ${m.layers} шари → голова`],
        ['Кубіти', String(m.n)],
        ['Масштаб γ', String(m.gamma)],
      ]
  rows.push(['Зерно (seed)', String(m.seed)], ['Параметри', m.params.toLocaleString('uk-UA')], ['F1 на тесті', m.test_f1.toFixed(3)], ['Точність', (m.test_acc * 100).toFixed(1) + ' %'])
  return <dl className="kv">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
}

function Result({ p, truth }: { p: Prediction & { at?: Date }; truth?: string }) {
  const t = p.timings_ms
  const stages = STAGES.filter((s) => t[s.key] !== undefined)
  return (
    <div className="result">
      <div className="verdict">
        <div className="big">{uk(p.label)}</div>
        <div className="muted">{p.label} · впевненість {pct(p.confidence)} · {modelName(p.model)}, seed {p.model.seed}</div>
        {truth && <div className={truth === p.label ? 'ok' : 'bad'}>{truth === p.label ? '✓ збігається зі справжнім класом' : `✗ справжній клас: ${uk(truth)}`}</div>}
      </div>
      <h3 className="sub">Найімовірніші класи</h3>
      <ul className="bars">
        {p.top.map((x) => (
          <li key={x.label}>
            <span className="lbl">{uk(x.label)}</span>
            <span className="track"><span className="fill" style={{ width: `${Math.max(x.p * 100, 0.5)}%` }} /></span>
            <span className="val">{pct(x.p)}</span>
          </li>
        ))}
      </ul>
      <h3 className="sub">Час етапів: {fmtMs(t.total)}{p.at && <span className="stamp"> · обчислено на сервері о {p.at.toLocaleTimeString('uk-UA')}</span>}</h3>
      <div className="stack" role="img" aria-label="Розподіл часу за етапами">
        {stages.map((s) => <span key={s.key} className={s.cls} style={{ flexGrow: t[s.key]! }} title={`${s.name}: ${fmtMs(t[s.key]!)}`} />)}
      </div>
      <ul className="legend">
        {stages.map((s) => <li key={s.key}><i className={s.cls} />{s.name}: {fmtMs(t[s.key]!)}</li>)}
      </ul>
      {p.model.kind === 'hybrid' && <p className="hint">Квантову схему виконано {p.device === 'braket.local.qubit' ? 'в локальному симуляторі Amazon Braket (braket.local.qubit)' : 'у вбудованому симуляторі PennyLane (default.qubit)'} на CPU, {p.shots ? `за ${p.shots} вимірюваннями (shots), тому ймовірності трохи змінюються від запуску до запуску` : 'точно (без вибірки вимірювань)'}. Її час не відображає швидкість реального квантового процесора.</p>}
    </div>
  )
}

function CompareTable({ c, truth, onPick }: { c: Comparison; truth?: string; onPick: (id: string) => void }) {
  const hy = c.results.find((r) => r.device)
  return (
    <div>
      <p className="hint">Ознаки YOLO11 обчислено один раз: {fmtMs(c.backbone_ms)}. Далі для кожної моделі показано лише час голови.{hy && <> Квантові схеми виконано {hy.device === 'braket.local.qubit' ? 'в локальному симуляторі Amazon Braket (braket.local.qubit)' : 'у вбудованому симуляторі PennyLane (default.qubit)'}{hy.shots ? `, ${hy.shots} shots` : ', точно'}.</>}</p>
      <div className="tbl-wrap">
        <table className="tbl">
          <thead><tr><th>Модель</th><th>Seed</th><th>Клас</th><th>Впевн.</th><th>Голова</th><th>F1 тест</th></tr></thead>
          <tbody>
            {c.results.map((r) => {
              const head = r.timings_ms.total
              return (
                <tr key={r.model.id} onClick={() => onPick(r.model.id)} className={truth ? (truth === r.label ? 'ok-row' : 'bad-row') : ''}>
                  <td>{modelName(r.model)}</td>
                  <td>{r.model.seed}</td>
                  <td>{uk(r.label)}</td>
                  <td>{pct(r.confidence)}</td>
                  <td>{fmtMs(head)}</td>
                  <td>{r.model.test_f1.toFixed(3)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const SHOTS = [0, 100, 1000, 10000]

/** Device for the quantum circuit of a hybrid model. Only free local simulators are offered. */
function DevicePicker({ devices, value, onChange, classical }: { devices: Device[]; value: RunOn; onChange: (v: RunOn) => void; classical: boolean }) {
  const list = devices.length ? devices : [{ id: 'default.qubit', name: DEVICE_UK['default.qubit'], available: true, error: null }]
  const braketOff = devices.find((d) => d.id === 'braket.local.qubit' && !d.available)
  return (
    <div className="device-pick">
      <label className="field">
        <span>Симулятор квантової схеми</span>
        <select value={value.device} disabled={classical}
          onChange={(e) => onChange({ device: e.target.value, shots: e.target.value === 'default.qubit' ? value.shots : value.shots || 1000 })}>
          {list.map((d) => <option key={d.id} value={d.id} disabled={!d.available}>{DEVICE_UK[d.id] ?? d.name}{d.available ? '' : ' (не встановлено)'}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Вимірювання (shots)</span>
        <select value={value.shots} disabled={classical} onChange={(e) => onChange({ ...value, shots: +e.target.value })}>
          {SHOTS.map((s) => <option key={s} value={s}>{s === 0 ? 'точно, без вибірки' : s.toLocaleString('uk-UA')}</option>)}
        </select>
      </label>
      {classical && <p className="hint">Логістична регресія не має квантової схеми, тому симулятор для неї не застосовується.</p>}
      {braketOff && <p className="hint">Щоб увімкнути Amazon Braket, встановіть плагін: pip install amazon-braket-pennylane-plugin</p>}
    </div>
  )
}
