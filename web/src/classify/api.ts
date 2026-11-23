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
  custom?: boolean   // trained on the «Тренування» page
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
  prepared?: string | null  // the image as the model saw it, when it was prepared (e.g. as an MNIST digit)
  device?: string | null    // where the circuit ran (null for the classical model)
  shots?: number | null     // null: exact expectation values
}
export interface Device { id: string; name: string; available: boolean; error: string | null }
/** Where the hybrid model's circuit runs: a local simulator, exactly (shots = 0) or with sampling. */
export interface RunOn { device: string; shots: number }
export const EXACT: RunOn = { device: 'default.qubit', shots: 0 }
export interface Comparison { backbone_ms: number; results: Prediction[]; prepared?: string | null }

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let msg = `Помилка сервера (${r.status}).`
    try { const j = await r.json(); if (typeof j.detail === 'string') msg = j.detail } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

const q = (dataset: string) => `dataset=${encodeURIComponent(dataset)}`
export const getDevices = () => call<Device[]>('/api/devices')
export const getModels = (dataset = 'voc') => call<Registry>(`/api/models?${q(dataset)}`)
export const getSamples = (dataset = 'voc') => call<Sample[]>(`/api/samples?${q(dataset)}`)
export const sampleUrl = (name: string, dataset = 'voc') => `/api/samples/${encodeURIComponent(name)}?${q(dataset)}`

export function classify(file: Blob, model: string, topK = 5, dataset = 'voc', prep = '', on: RunOn = EXACT) {
  const fd = new FormData()
  fd.append('device', on.device)
  fd.append('shots', String(on.shots))
  fd.append('file', file, 'image')
  fd.append('model', model)
  fd.append('top_k', String(topK))
  fd.append('dataset', dataset)
  fd.append('prep', prep)
  return call<Prediction>('/api/classify', { method: 'POST', body: fd })
}

export function compareAll(file: Blob, topK = 3, dataset = 'voc', prep = '', on: RunOn = EXACT) {
  const fd = new FormData()
  fd.append('device', on.device)
  fd.append('shots', String(on.shots))
  fd.append('file', file, 'image')
  fd.append('top_k', String(topK))
  fd.append('dataset', dataset)
  fd.append('prep', prep)
  return call<Comparison>('/api/classify/compare', { method: 'POST', body: fd })
}
