// Thin client for the FastAPI backend (server/app.py). In dev, Vite proxies /api to :8000.

export interface ModelInfo {
  id: string
  kind: 'classical' | 'hybrid'
  encoding: 'amplitude' | 'angle' | 'zz' | null
  n: number | null
  gamma: number | null
  seed: number
  layers: number
  params: number
  val_f1: number
  test_f1: number
  test_acc: number
  train_s: number
}

export interface Registry { dataset: string; classes: string[]; models: ModelInfo[] }
export interface Sample { name: string; label: string }
export interface Timings { preprocess: number; backbone: number; adapter?: number; circuit?: number; head: number; total: number }
export interface Prediction {
  model: ModelInfo
  label: string
  confidence: number
  top: { label: string; p: number }[]
  timings_ms: Timings
}
export interface Comparison { backbone_ms: number; results: Prediction[] }

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let msg = `Помилка сервера (${r.status}).`
    try { const j = await r.json(); if (typeof j.detail === 'string') msg = j.detail } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export const getModels = () => call<Registry>('/api/models')
export const getSamples = () => call<Sample[]>('/api/samples')
export const sampleUrl = (name: string) => `/api/samples/${encodeURIComponent(name)}`

export function classify(file: Blob, model: string, topK = 5) {
  const fd = new FormData()
  fd.append('file', file, 'image')
  fd.append('model', model)
  fd.append('top_k', String(topK))
  return call<Prediction>('/api/classify', { method: 'POST', body: fd })
}

export function compareAll(file: Blob, topK = 3) {
  const fd = new FormData()
  fd.append('file', file, 'image')
  fd.append('top_k', String(topK))
  return call<Comparison>('/api/classify/compare', { method: 'POST', body: fd })
}
