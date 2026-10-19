import { useEffect, useState } from 'react'
import App from './App'
import { ClassifyPage } from './classify/ClassifyPage'

type Tab = 'composer' | 'classify'
const fromHash = (): Tab => (location.hash === '#classify' ? 'classify' : 'composer')

/** App shell: switches between the circuit composer and the classification page. Both stay mounted to keep their state. */
export default function Root() {
  const [tab, setTab] = useState<Tab>(fromHash)
  useEffect(() => {
    const on = () => setTab(fromHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const go = (t: Tab) => { history.replaceState(null, '', t === 'classify' ? '#classify' : '#'); setTab(t) }

  return (
    <>
      <nav className="tabs" role="tablist">
        <span className="brand">QHybrid-CV</span>
        <button role="tab" aria-selected={tab === 'composer'} className={tab === 'composer' ? 'on' : ''} onClick={() => go('composer')}>Конструктор схем</button>
        <button role="tab" aria-selected={tab === 'classify'} className={tab === 'classify' ? 'on' : ''} onClick={() => go('classify')}>Класифікація</button>
      </nav>
      <div hidden={tab !== 'composer'}><App /></div>
      <div hidden={tab !== 'classify'}><ClassifyPage /></div>
    </>
  )
}
