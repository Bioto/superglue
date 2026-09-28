import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { api, clearMasterKey, getMasterKey, setMasterKey } from './api'
import { GatewayBudgetsPage } from './pages/gateway/BudgetsPage'
import { GatewayCapturePage } from './pages/gateway/CapturePage'
import { GatewayCaptureRecordDetailPage } from './pages/gateway/CaptureRecordDetailPage'
import { GatewayCaptureRecordsPage } from './pages/gateway/CaptureRecordsPage'
import { GatewayKeysPage } from './pages/gateway/KeysPage'
import { GatewayOverviewPage } from './pages/gateway/OverviewPage'
import { GatewayProvidersPage } from './pages/gateway/ProvidersPage'
import { GatewayUsagePage } from './pages/gateway/UsagePage'
import { GatewayUsersPage } from './pages/gateway/UsersPage'
import { GatewayNavProvider, type GatewayCaptureView, type GatewayNav } from './components/agent/gatewayNav'
import './style.css'

type Tab = 'overview' | 'users' | 'keys' | 'budgets' | 'usage' | 'providers' | 'capture'

function Login({ onLogin }: { onLogin: () => void }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setLoading(true)
    setMasterKey(key.trim())
    try {
      await api.users()
      onLogin()
    } catch (cause) {
      clearMasterKey()
      setError(cause instanceof Error ? cause.message : 'The master key was rejected.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-shell">
      <form className="login-card" onSubmit={submit}>
        <p className="eyebrow">SUPERGLUE</p>
        <h1>Gateway admin</h1>
        <p className="muted">Use the gateway master key to manage users, keys, budgets, and usage.</p>
        <label>
          Master key
          <input
            autoFocus
            type="password"
            value={key}
            onChange={(event) => setKey(event.target.value)}
            placeholder="GATEWAY_MASTER_KEY"
          />
        </label>
        {error ? <p className="error">{error}</p> : null}
        <button disabled={!key.trim() || loading} type="submit">{loading ? 'Checking…' : 'Sign in'}</button>
      </form>
    </main>
  )
}

function AdminApp({ onLogout }: { onLogout: () => void }) {
  const [tab, setTab] = useState<Tab>('overview')
  const [captureView, setCaptureView] = useState<'status' | 'records' | 'detail'>('status')
  const [requestId, setRequestId] = useState<string>()
  const tabs: [Tab, string][] = [
    ['overview', 'Overview'],
    ['users', 'Users'],
    ['keys', 'Keys'],
    ['budgets', 'Budgets'],
    ['usage', 'Usage'],
    ['providers', 'Providers'],
    ['capture', 'Capture'],
  ]

  const openCapture = (view: 'status' | 'records' | 'detail', id?: string) => {
    setTab('capture')
    setCaptureView(view)
    setRequestId(id)
  }
  const nav: GatewayNav = {
    openTab: (next) => {
      setTab(next)
      if (next === 'capture') setCaptureView('status')
    },
    openCapture: (view: GatewayCaptureView) => openCapture(view.name, view.name === 'detail' ? view.requestId : undefined),
  }

  return (
    <div className="admin-shell">
      <header className="admin-header">
        <div><p className="eyebrow">SUPERGLUE</p><h1>Gateway admin</h1></div>
        <button className="secondary" onClick={onLogout} type="button">Sign out</button>
      </header>
      <nav className="tabs" aria-label="Gateway admin">
        {tabs.map(([id, label]) => (
          <button className={tab === id ? 'active' : ''} key={id} onClick={() => { setTab(id); if (id === 'capture') setCaptureView('status') }} type="button">
            {label}
          </button>
        ))}
      </nav>
      <GatewayNavProvider value={nav}>
        <main className="admin-main">
        {tab === 'overview' ? <GatewayOverviewPage /> : null}
        {tab === 'users' ? <GatewayUsersPage /> : null}
        {tab === 'keys' ? <GatewayKeysPage /> : null}
        {tab === 'budgets' ? <GatewayBudgetsPage /> : null}
        {tab === 'usage' ? <GatewayUsagePage /> : null}
        {tab === 'providers' ? <GatewayProvidersPage /> : null}
        {tab === 'capture' && captureView === 'status' ? <GatewayCapturePage /> : null}
        {tab === 'capture' && captureView === 'records' ? (
          <GatewayCaptureRecordsPage />
        ) : null}
        {tab === 'capture' && captureView === 'detail' ? (
          <GatewayCaptureRecordDetailPage requestId={requestId} />
        ) : null}
        </main>
      </GatewayNavProvider>
    </div>
  )
}

function App() {
  const [authenticated, setAuthenticated] = useState(Boolean(getMasterKey()))
  useEffect(() => {
    if (!authenticated) clearMasterKey()
  }, [authenticated])
  return authenticated
    ? <AdminApp onLogout={() => { clearMasterKey(); setAuthenticated(false) }} />
    : <Login onLogin={() => setAuthenticated(true)} />
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
