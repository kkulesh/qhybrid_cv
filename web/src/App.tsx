import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CircuitCanvas } from './components/CircuitCanvas'
import { CodePanel } from './components/CodePanel'
import { Inspector } from './components/Inspector'
import { Palette } from './components/Palette'
import { ResultsPanel } from './components/ResultsPanel'
import { GATES, defaultParams, newId, type Circuit, type GateKind, type Op } from './quantum/gates'
import { compact, depth } from './quantum/layout'
import { simulate } from './quantum/simulator'
import { TEMPLATES } from './quantum/templates'
import { download } from './util'

const MAX_QUBITS = 10
const STORAGE_KEY = 'qhybrid-composer-v1'

interface Doc {
  title: string
  circuit: Circuit
}

function loadInitial(): Doc {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw) as Doc
  } catch { /* storage unavailable */ }
  return { title: 'Стан Белла', circuit: TEMPLATES[1].make() }
}

/** Fit a gate's wires around qubit q so that it stays inside the register. */
function placeWires(arity: number, q: number, n: number): number[] | null {
  if (arity > n) return null
  const start = Math.min(q, n - arity)
  const wires = [...Array(arity).keys()].map((i) => start + i)
  // keep the clicked qubit as the first (control) wire
  return [q, ...wires.filter((w) => w !== q)].slice(0, arity)
}

export default function App() {
  const [doc, setDoc] = useState<Doc>(loadInitial)
  const [past, setPast] = useState<Doc[]>([])
  const [future, setFuture] = useState<Doc[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [armed, setArmed] = useState<GateKind | null>(null)
  const [theme, setTheme] = useState<'light' | 'dark' | 'auto'>('auto')
  const [fileError, setFileError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const { circuit } = doc

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(doc)) } catch { /* ignore */ }
  }, [doc])
  useEffect(() => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const commit = useCallback((next: Doc) => {
    setPast((p) => [...p.slice(-99), doc])
    setFuture([])
    setDoc(next)
  }, [doc])
  const setCircuit = (c: Circuit) => commit({ ...doc, circuit: { ...c, ops: compact(c.ops, c.nQubits) } })

  const undo = useCallback(() => {
    if (!past.length) return
    setFuture((f) => [doc, ...f])
    setDoc(past[past.length - 1])
    setPast((p) => p.slice(0, -1))
  }, [past, doc])
  const redo = useCallback(() => {
    if (!future.length) return
    setPast((p) => [...p, doc])
    setDoc(future[0])
    setFuture((f) => f.slice(1))
  }, [future, doc])

  const removeOp = useCallback((id: string) => {
    commit({ ...doc, circuit: { ...circuit, ops: compact(circuit.ops.filter((o) => o.id !== id), circuit.nQubits) } })
    setSelected(null)
  }, [commit, doc, circuit])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement).closest('input, textarea, select')
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !typing) { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y' && !typing) { e.preventDefault(); redo() }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selected && !typing) removeOp(selected)
      else if (e.key === 'Escape') { setArmed(null); setSelected(null) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undo, redo, removeOp, selected])

  const sim = useMemo(() => simulate(circuit), [circuit])
  const selectedOp = circuit.ops.find((o) => o.id === selected) ?? null

  const dropGate = (kind: GateKind, q: number, col: number) => {
    const wires = placeWires(GATES[kind].arity, q, circuit.nQubits)
    if (!wires) return
    const op: Op = { id: newId(), kind, wires, params: defaultParams(kind), col: col - 0.5 }
    setCircuit({ ...circuit, ops: [...circuit.ops, op] })
    setSelected(op.id)
  }
  const moveOp = (id: string, dq: number, col: number) => {
    const op = circuit.ops.find((o) => o.id === id)
    if (!op) return
    const wires = op.wires.map((w) => w + dq)
    if (wires.some((w) => w < 0 || w >= circuit.nQubits)) return
    const target = col < op.col ? col - 0.5 : col + 0.5
    setCircuit({ ...circuit, ops: circuit.ops.map((o) => (o.id === id ? { ...o, wires, col: target } : o)) })
  }
  const setQubits = (n: number) => {
    if (n < 1 || n > MAX_QUBITS) return
    setCircuit({ nQubits: n, ops: circuit.ops.filter((o) => o.wires.every((w) => w < n)) })
  }

  return (
    <div className="app">
      <header className="topbar">
        <input className="title" value={doc.title} aria-label="Назва схеми" onChange={(e) => setDoc({ ...doc, title: e.target.value })} />
        <nav className="menu">
          <select aria-label="Шаблони" value="" onChange={(e) => {
            const t = TEMPLATES.find((x) => x.key === e.target.value)
            if (t) { commit({ title: t.name, circuit: t.make() }); setSelected(null) }
          }}>
            <option value="" disabled>Шаблони…</option>
            {TEMPLATES.map((t) => <option key={t.key} value={t.key}>{t.name}</option>)}
          </select>
          <button className="btn" onClick={() => fileRef.current?.click()}>Відкрити</button>
          <button className="btn" onClick={() => download(`${doc.title || 'circuit'}.json`, JSON.stringify(doc, null, 2), 'application/json')}>Зберегти</button>
          <button className="btn" onClick={undo} disabled={!past.length} title="Ctrl+Z">Скасувати</button>
          <button className="btn" onClick={redo} disabled={!future.length} title="Ctrl+Y">Повторити</button>
          <button className="btn" onClick={() => { setCircuit({ ...circuit, ops: [] }); setSelected(null) }}>Очистити</button>
          <select aria-label="Тема" value={theme} onChange={(e) => setTheme(e.target.value as typeof theme)}>
            <option value="auto">Тема: системна</option>
            <option value="light">Тема: світла</option>
            <option value="dark">Тема: темна</option>
          </select>
        </nav>
        {fileError && <span className="error-note" role="alert">{fileError}</span>}
        <input ref={fileRef} type="file" accept=".json,application/json" hidden onChange={async (e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (!f) return
          try {
            const d = JSON.parse(await f.text()) as Doc
            if (!d.circuit || typeof d.circuit.nQubits !== 'number' || !Array.isArray(d.circuit.ops)) throw new Error('format')
            commit({ title: d.title ?? f.name, circuit: { ...d.circuit, ops: compact(d.circuit.ops.filter((o) => o.kind in GATES), d.circuit.nQubits) } })
            setFileError(null)
          } catch { setFileError('Не вдалося прочитати файл: очікується JSON, збережений цим редактором.') }
        }} />
      </header>

      <main className="workspace">
        <Palette armed={armed} onArm={setArmed} />
        <section className="panel canvas-panel">
          <header className="panel-head">
            <span className="muted small">Кубітів:</span>
            <button className="btn icon" aria-label="Прибрати кубіт" onClick={() => setQubits(circuit.nQubits - 1)} disabled={circuit.nQubits <= 1}>−</button>
            <strong>{circuit.nQubits}</strong>
            <button className="btn icon" aria-label="Додати кубіт" onClick={() => setQubits(circuit.nQubits + 1)} disabled={circuit.nQubits >= MAX_QUBITS}>+</button>
            <span className="muted small">Глибина: {depth(circuit.ops)} · Вентилів: {circuit.ops.filter((o) => o.kind !== 'BARRIER').length}</span>
            {armed && <span className="armed-note">Обрано {GATES[armed].label}: натисніть на клітинку (Esc скасовує)</span>}
          </header>
          <CircuitCanvas circuit={circuit} selected={selected} armed={armed} onSelect={setSelected} onDropGate={dropGate} onMoveOp={moveOp} />
          <Inspector op={selectedOp} nQubits={circuit.nQubits} onDelete={removeOp} onChange={(op) => setCircuit({ ...circuit, ops: circuit.ops.map((o) => (o.id === op.id ? op : o)) })} />
        </section>
        <CodePanel circuit={circuit} title={doc.title} />
      </main>
      <ResultsPanel sim={sim} nQubits={circuit.nQubits} />
    </div>
  )
}
