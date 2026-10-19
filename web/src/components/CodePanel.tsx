import { useState } from 'react'
import type { Circuit } from '../quantum/gates'
import { toPennyLane, toQasm } from '../quantum/codegen'
import { download } from '../util'

export function CodePanel({ circuit, title }: { circuit: Circuit; title: string }) {
  const [lang, setLang] = useState<'qasm' | 'pennylane'>('qasm')
  const code = lang === 'qasm' ? toQasm(circuit) : toPennyLane(circuit)
  const [copied, setCopied] = useState(false)
  return (
    <section className="panel code">
      <header className="panel-head">
        <select value={lang} onChange={(e) => setLang(e.target.value as 'qasm' | 'pennylane')} aria-label="Мова коду">
          <option value="qasm">OpenQASM 3.0</option>
          <option value="pennylane">PennyLane (Python)</option>
        </select>
        <div className="spacer" />
        <button className="btn" onClick={async () => {
          try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1200) } catch { /* clipboard unavailable */ }
        }}>{copied ? 'Скопійовано' : 'Копіювати'}</button>
        <button className="btn" onClick={() => download(`${slug(title)}.${lang === 'qasm' ? 'qasm' : 'py'}`, code)}>Завантажити</button>
      </header>
      <pre className="codeview"><code>{code.split('\n').map((l, i) => (
        <span key={i} className="cline"><span className="ln">{i + 1}</span>{l}{'\n'}</span>
      ))}</code></pre>
    </section>
  )
}

const slug = (s: string) => s.trim().replace(/[^\p{L}\p{N}]+/gu, '_').replace(/^_|_$/g, '') || 'circuit'
