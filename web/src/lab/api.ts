// Client for the background jobs of server/app.py: encoding selection and model training.
// Progress arrives as Server-Sent Events from /api/jobs/{id}/events.
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ModelInfo } from '../classify/api'

export type Encoding = 'angle' | 'zz' | 'amplitude'
export const ENCODINGS: Encoding[] = ['angle', 'zz', 'amplitude']
export const GAMMAS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

export interface SelectionParams { dataset: string; n: number; per_class: number; gammas: number[]; encodings: Encoding[]; seed?: number }
export interface TrainParams { dataset: string; kind: 'hybrid' | 'classical'; encoding: Encoding; n: number; gamma: number; layers: number; epochs: number; lr: number; batch_size: number; per_class: number; seed: number }
export interface SelRow { encoding: Encoding; gamma: number; kta: number; var: number; hs: number; time_s: number }
export interface SelResult { rows: SelRow[]; best: SelRow; best_per_encoding: SelRow[]; sample_size: number; time_s: number }
export interface ImportResult { dataset: string; classes: string[]; n_train: number; n_test: number; skipped: number; split: string; time_s: number }
export interface TrainResult { model: ModelInfo & { custom?: boolean; epochs: number; per_class: number } }

export type JobEvent =
  | { type: 'row'; t: number; row: SelRow; done: number; total: number }
  | { type: 'start'; t: number; n_train: number; n_val: number; n_test: number; params: number }
  | { type: 'epoch'; t: number; epoch: number; epochs: number; loss: number; val_f1: number; elapsed_s: number }
  | { type: 'evaluating'; t: number }
  | { type: 'progress'; t: number; stage: 'unzip' | 'features'; done: number; total: number; elapsed_s?: number }
  | { type: 'layout'; t: number; classes: string[]; n_train: number; n_test: number; note: string }
  | { type: 'done'; t: number; result: SelResult | TrainResult | ImportResult }
  | { type: 'error'; t: number; message: string }
  | { type: 'cancelled'; t: number }

export type Kind = 'selection' | 'train' | 'dataset'
type Status = 'idle' | 'running' | 'done' | 'error' | 'cancelled'
interface JobSummary { id: string; kind: Kind; params: unknown; status: Status }

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let msg = `Помилка сервера (${r.status}).`
    try { const j = await r.json(); if (typeof j.detail === 'string') msg = j.detail; else if (Array.isArray(j.detail)) msg = 'Некоректні параметри: ' + j.detail.map((d: { loc: string[] }) => d.loc[d.loc.length - 1]).join(', ') } catch { /* not json */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

const post = <T,>(path: string, body?: unknown) =>
  call<T>(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })


/**
 * One job of the given kind: start it, follow its events, cancel it. On mount it reattaches to the
 * latest job of this kind on the server, so a page reload does not lose a running training.
 */
export function useJob<P>(kind: Kind) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [params, setParams] = useState<P | null>(null)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const es = useRef<EventSource | null>(null)

  const follow = useCallback((id: string, current: Status = 'running') => {
    es.current?.close()
    setJobId(id); setEvents([]); setStatus(current); setError(null)
    const src = new EventSource(`/api/jobs/${id}/events`)
    es.current = src
    src.onmessage = (m) => {
      const ev = JSON.parse(m.data) as JobEvent
      setEvents((old) => [...old, ev])
      if (ev.type === 'done' || ev.type === 'error' || ev.type === 'cancelled') {
        setStatus(ev.type)
        if (ev.type === 'error') setError(ev.message)
        src.close()
      }
    }
    src.onerror = () => {
      if (src.readyState === EventSource.CLOSED) { setStatus('error'); setError("З'єднання з сервером втрачено.") }
    }
  }, [])

  useEffect(() => {
    call<JobSummary[]>('/api/jobs')
      .then((list) => { const j = list.find((x) => x.kind === kind); if (j) { setParams(j.params as P); follow(j.id, j.status) } })
      .catch(() => { /* the page shows the offline note from getFeatures */ })
    return () => es.current?.close()
  }, [kind, follow])

  const start = useCallback(async (p: P) => {
    setError(null)
    try {
      const { id } = await post<{ id: string }>(kind === 'selection' ? '/api/selection' : '/api/train', p)
      setParams(p); follow(id)
    } catch (e) { setError((e as Error).message) }
  }, [kind, follow])

  const cancel = useCallback(() => { if (jobId) post(`/api/jobs/${jobId}/cancel`).catch(() => {}) }, [jobId])

  /** Follow a job started elsewhere (the dataset upload posts a multipart form itself). */
  const attach = useCallback((id: string, p: P) => { setParams(p); follow(id) }, [follow])

  return { jobId, params, events, status, error, start, cancel, attach }
}

/** Parameters passed in the address, e.g. #train?enc=amplitude&n=8&gamma=1 */
export function hashParams(): URLSearchParams {
  const i = location.hash.indexOf('?')
  return new URLSearchParams(i < 0 ? '' : location.hash.slice(i + 1))
}

export const lastOf = <T,>(a: T[]): T | undefined => a[a.length - 1]
