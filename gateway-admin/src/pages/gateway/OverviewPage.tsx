import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../../api'
import { GatewayNavButton } from '../../components/agent/gatewayNav'
import type { GatewayHealth, UsageSummary, CaptureStatus } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

function MiniBarChart({ groups }: { groups: UsageSummary['groups'] }) {
  const max = Math.max(...groups.map((g) => g.cost_usd), 0.0001)
  return (
    <div className="bar-chart">
      {groups.slice(0, 14).reverse().map((g) => (
        <div key={g.key} className="bar-row" title={`${g.key}: $${g.cost_usd.toFixed(4)}`}>
          <span className="bar-label mono">{g.key.slice(5, 10)}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(g.cost_usd / max) * 100}%` }} />
          </div>
          <span className="bar-value">${g.cost_usd.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

function StatCard({ label, value, children }: { label: string; value?: string | number; children?: ReactNode }) {
  return (
    <Card size="sm">
      <CardContent className="grid gap-1">
        <div className="text-xs text-muted-foreground">{label}</div>
        {value !== undefined && <div className="font-mono text-base font-semibold">{value}</div>}
        {children}
      </CardContent>
    </Card>
  )
}

export function GatewayOverviewPage() {
  const [health, setHealth] = useState<GatewayHealth | null>(null)
  const [counts, setCounts] = useState({ users: 0, keys: 0, budgets: 0 })
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [daily, setDaily] = useState<UsageSummary | null>(null)
  const [capture, setCapture] = useState<CaptureStatus | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(console.error)
    Promise.all([
      api.users().catch(() => ({ users: [] })),
      api.keys().catch(() => ({ keys: [] })),
      api.budgets().catch(() => ({ budgets: [] })),
      api.usageSummary({ group_by: 'user' }).catch(() => null),
      api.usageSummary({ group_by: 'day' }).catch(() => null),
      api.captureStatus().catch(() => null),
    ]).then(([u, k, b, s, d, c]) => {
      setCounts({ users: u.users.length, keys: k.keys.length, budgets: b.budgets.length })
      if (s) setSummary(s)
      if (d) setDaily(d)
      if (c) setCapture(c)
    })
  }, [])

  if (!health) return <div className="p-4 text-sm text-muted-foreground">Loading gateway status…</div>

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Gateway</h2>
      </div>
      {!health.configured && (
        <Alert variant="destructive"><AlertDescription>{health.error ?? 'Gateway not configured'}</AlertDescription></Alert>
      )}
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Configured" value={health.configured ? 'Yes' : 'No'} />
        <StatCard label="Health" value={health.live ? 'Live' : 'Down'} />
        <StatCard label="Ready" value={health.ready ? 'Ready' : 'Not ready'} />
        <StatCard label="URL" value={health.gateway_url ?? '—'} />
      </div>
      {summary && (
        <div className="grid gap-2 sm:grid-cols-3">
          <StatCard label="Total spend" value={`$${summary.totals.cost_usd.toFixed(2)}`} />
          <StatCard label="Requests" value={summary.totals.requests} />
          <StatCard label="Tokens" value={(summary.totals.prompt_tokens + summary.totals.completion_tokens).toLocaleString()} />
        </div>
      )}
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <StatCard label="Users" value={counts.users}>
          <GatewayNavButton>Manage →</GatewayNavButton>
        </StatCard>
        <StatCard label="Keys" value={counts.keys}>
          <GatewayNavButton>Manage →</GatewayNavButton>
        </StatCard>
        <StatCard label="Budgets" value={counts.budgets}>
          <GatewayNavButton>Manage →</GatewayNavButton>
        </StatCard>
        <StatCard label="Providers">
          <GatewayNavButton>View status →</GatewayNavButton>
        </StatCard>
        <StatCard label="Capture" value={capture?.enabled ? 'On' : 'Off'}>
          {capture?.enabled && capture.stats && capture.stats.dropped_records > 0 && (
            <div className="text-xs text-muted-foreground">{capture.stats.dropped_records} dropped</div>
          )}
          <GatewayNavButton>Control panel →</GatewayNavButton>
        </StatCard>
      </div>
      {daily && daily.groups.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Daily cost (recent)</CardTitle></CardHeader>
          <CardContent><MiniBarChart groups={daily.groups} /></CardContent>
        </Card>
      )}
    </div>
  )
}
