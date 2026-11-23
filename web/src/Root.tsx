import { useEffect, useState } from 'react'
import App from './App'
import { ClassifyPage } from './classify/ClassifyPage'
import { DatasetPage } from './datasets/DatasetPage'
import { DatasetProvider } from './datasets/datasets'
import { SelectionPage } from './lab/SelectionPage'
import { TrainPage } from './lab/TrainPage'

type Tab = 'composer' | 'data' | 'select' | 'train' | 'classify'
const TABS: { id: Tab; name: string }[] = [
  { id: 'composer', name: 'Конструктор схем' },
  { id: 'data', name: 'Набір даних' },
  { id: 'select', name: 'Вибір кодування' },
  { id: 'train', name: 'Тренування' },
  { id: 'classify', name: 'Класифікація' },
]
// the hash may carry parameters, e.g. #train?enc=amplitude&n=8&gamma=1
const fromHash = (): Tab => TABS.find((t) => t.id !== 'composer' && location.hash.split('?')[0] === `#${t.id}`)?.id ?? 'composer'

/** App shell: switches between the pages. All stay mounted to keep their state. */
export default function Root() {
  const [tab, setTab] = useState<Tab>(fromHash)
  useEffect(() => {
    const on = () => setTab(fromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const go = (t: Tab) => { history.replaceState(null, '', t === 'composer' ? '#' : `#${t}`); setTab(t) }

  return (
    <DatasetProvider>
      <nav className="tabs" role="tablist">
        <span className="brand">QHybrid-CV</span>
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? 'on' : ''} onClick={() => go(t.id)}>{t.name}</button>
        ))}
      </nav>
      <div hidden={tab !== 'composer'}><App /></div>
      <div hidden={tab !== 'data'}><DatasetPage /></div>
      <div hidden={tab !== 'select'}><SelectionPage /></div>
      <div hidden={tab !== 'train'}><TrainPage /></div>
      <div hidden={tab !== 'classify'}><ClassifyPage /></div>
    </DatasetProvider>
  )
}
