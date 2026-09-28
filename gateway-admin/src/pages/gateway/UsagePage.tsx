import { useEffect, useState } from 'react'
import { api } from '../../api'
import { compactTime, keyLabel, shortId, userLabel } from '../../components/compactDisplay'
import { KeySelect, UserSelect } from '../../components/GatewayEntitySelect'
import { PaginatedTable } from '../../components/PaginatedTable'
import type { GatewayKey, GatewayUsage, GatewayUser, UsageSummary } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

export function GatewayUsagePage() {
  const [usage, setUsage] = useState<GatewayUsage[]>([])
  const [summary, setSummary] = useState<UsageSummary | null>(null)
  const [byModel, setByModel] = useState<UsageSummary | null>(null)
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [keys, setKeys] = useState<GatewayKey[]>([])
  const [error, setError] = useState<string | null>(null)
  const [userId, setUserId] = useState('')
  const [keyId, setKeyId] = useState('')
  const [limit, setLimit] = useState(100)

  const userById = Object.fromEntries(users.map((user) => [user.id, user]))
  const keyById = Object.fromEntries(keys.map((key) => [key.id, key]))

  const load = () => {
    const params = { user_id: userId || undefined, key_id: keyId || undefined, limit }
    Promise.all([
      api.usage(params),
      api.usageSummary({ ...params, group_by: 'user' }),
      api.usageSummary({ ...params, group_by: 'model' }),
    ])
      .then(([u, s, m]) => {
        setUsage(u.usage)
        setSummary(s)
        setByModel(m)
      })
      .catch((e) => setError(String(e)))
  }

  useEffect(() => {
    Promise.all([api.users(), api.keys()])
      .then(([u, k]) => {
        setUsers(u.users)
        setKeys(k.keys)
      })
      .catch((e) => setError(String(e)))
    load()
  }, [])

  const selectUser = (id: string) => {
    setUserId(id)
    const selected = keys.find((key) => key.id === keyId)
    if (selected && id && selected.user_id !== id) setKeyId('')
  }

  const selectKey = (id: string) => {
    setKeyId(id)
    if (!id || userId) return
    const selected = keys.find((key) => key.id === id)
    if (selected) setUserId(selected.user_id)
  }

  return (
    <div className="grid gap-3">
      <h2 className="text-lg font-semibold">Gateway usage</h2>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      <Card>
        <CardContent className="flex flex-wrap items-end gap-3">
          <UserSelect id="usage-user" users={users} value={userId} onChange={selectUser} allowAll />
          <KeySelect id="usage-key" keys={keys} users={users} userId={userId} value={keyId} onChange={selectKey} />
          <div className="grid w-24 gap-1.5"><Label htmlFor="usage-limit">Limit</Label><Input id="usage-limit" type="number" value={limit} onChange={(e) => setLimit(Number(e.target.value))} /></div>
          <Button onClick={load}>Refresh</Button>
        </CardContent>
      </Card>
      {summary && (
        <div className="grid gap-2 sm:grid-cols-3">
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Total cost</div><div className="font-mono text-base font-semibold">${summary.totals.cost_usd.toFixed(4)}</div></CardContent></Card>
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Requests</div><div className="font-mono text-base font-semibold">{summary.totals.requests}</div></CardContent></Card>
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Tokens</div><div className="font-mono text-base font-semibold">{(summary.totals.prompt_tokens + summary.totals.completion_tokens).toLocaleString()}</div></CardContent></Card>
        </div>
      )}
      {byModel && byModel.groups.length > 0 && (
        <Card>
          <CardHeader><CardTitle>By model</CardTitle></CardHeader>
          <CardContent>
            <Table>
              <TableHeader><TableRow><TableHead>Model</TableHead><TableHead>Requests</TableHead><TableHead>Cost</TableHead></TableRow></TableHeader>
              <TableBody>{byModel.groups.slice(0, 10).map((g) => (
                <TableRow key={g.key}><TableCell title={g.key}>{g.key}</TableCell><TableCell>{g.requests}</TableCell><TableCell>${g.cost_usd.toFixed(4)}</TableCell></TableRow>
              ))}</TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      <PaginatedTable
        rows={usage}
        searchKeys={(u) => [
          userLabel(userById[u.user_id], u.user_id),
          u.user_id,
          u.model,
          u.request_id ?? '',
          keyLabel(u.key_id ? keyById[u.key_id] : undefined, u.key_id),
          u.key_id ?? '',
        ]}
        searchPlaceholder="Search usage…"
      >
        {(pageRows) => (
          <Table>
            <TableHeader><TableRow><TableHead>Time</TableHead><TableHead>User</TableHead><TableHead>Key</TableHead><TableHead>Model</TableHead><TableHead>Tokens</TableHead><TableHead>Cost</TableHead><TableHead>Request ID</TableHead></TableRow></TableHeader>
            <TableBody>
              {pageRows.length === 0 ? (
                <TableRow><TableCell colSpan={7} className="text-center text-muted-foreground">No usage records.</TableCell></TableRow>
              ) : pageRows.map((u) => (
                <TableRow key={u.id}>
                  <TableCell className="font-mono" title={u.created_at}>{compactTime(u.created_at)}</TableCell>
                  <TableCell title={u.user_id}>{userLabel(userById[u.user_id], u.user_id)}</TableCell>
                  <TableCell title={u.key_id ?? undefined}>{u.key_id ? keyLabel(keyById[u.key_id], u.key_id) : '—'}</TableCell>
                  <TableCell title={u.model}>{u.model}</TableCell>
                  <TableCell>{u.prompt_tokens}+{u.completion_tokens}</TableCell>
                  <TableCell>${u.cost_usd.toFixed(4)}</TableCell>
                  <TableCell className="font-mono" title={u.request_id ?? undefined}>{u.request_id ? shortId(u.request_id) : '—'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </PaginatedTable>
    </div>
  )
}
