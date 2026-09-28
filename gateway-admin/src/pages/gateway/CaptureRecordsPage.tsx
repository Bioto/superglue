import { useEffect, useState } from 'react'
import { api } from '../../api'
import { GatewayNavButton, useGatewayNav } from '../../components/agent/gatewayNav'
import { compactTime, shortId, userLabel } from '../../components/compactDisplay'
import { UserSelect } from '../../components/GatewayEntitySelect'
import type { CaptureRecordSummary, GatewayUser } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

function defaultFrom(): string { return new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().slice(0, 16) }
function defaultTo(): string { return new Date().toISOString().slice(0, 16) }
function toRfc3339(local: string): string | undefined { return local ? new Date(local).toISOString() : undefined }

export function GatewayCaptureRecordsPage() {
  const { openCapture } = useGatewayNav()
  const [records, setRecords] = useState<CaptureRecordSummary[]>([])
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [error, setError] = useState<string | null>(null)
  const [userId, setUserId] = useState('')
  const [requestId, setRequestId] = useState('')
  const [model, setModel] = useState('')
  const [from, setFrom] = useState(defaultFrom)
  const [to, setTo] = useState(defaultTo)
  const [limit, setLimit] = useState(50)
  const [cursor, setCursor] = useState<string | null>(null)
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = (append = false) => {
    setLoading(true)
    setError(null)
    api.captureRecords({ user_id: userId || undefined, request_id: requestId || undefined, model: model || undefined, from: toRfc3339(from), to: toRfc3339(to), limit, cursor: append ? cursor ?? undefined : undefined })
      .then((res) => {
        setRecords((prev) => (append ? [...prev, ...res.records] : res.records))
        setNextCursor(res.truncated ? res.next_cursor ?? null : null)
        if (!append) setCursor(null)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }

  const loadMore = () => {
    if (!nextCursor) return
    setCursor(nextCursor)
    setLoading(true)
    setError(null)
    api.captureRecords({ user_id: userId || undefined, request_id: requestId || undefined, model: model || undefined, from: toRfc3339(from), to: toRfc3339(to), limit, cursor: nextCursor })
      .then((res) => { setRecords((prev) => [...prev, ...res.records]); setNextCursor(res.truncated ? res.next_cursor ?? null : null) })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    api.users().then((u) => setUsers(u.users)).catch((e) => setError(String(e)))
    load(false)
  }, [])

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3"><h2 className="text-lg font-semibold">Captured records</h2><GatewayNavButton capture={{ name: 'status' }}>Control panel</GatewayNavButton></div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      <Card><CardContent className="flex flex-wrap items-end gap-3">
        <UserSelect id="capture-user" users={users} value={userId} onChange={setUserId} allowAll />
        <div className="grid min-w-40 flex-1 gap-1.5"><Label htmlFor="capture-request">Request ID</Label><Input id="capture-request" value={requestId} onChange={(e) => setRequestId(e.target.value)} /></div>
        <div className="grid min-w-40 flex-1 gap-1.5"><Label htmlFor="capture-model">Model</Label><Input id="capture-model" value={model} onChange={(e) => setModel(e.target.value)} /></div>
        <div className="grid gap-1.5"><Label htmlFor="capture-from">From</Label><Input id="capture-from" type="datetime-local" value={from} onChange={(e) => setFrom(e.target.value)} /></div>
        <div className="grid gap-1.5"><Label htmlFor="capture-to">To</Label><Input id="capture-to" type="datetime-local" value={to} onChange={(e) => setTo(e.target.value)} /></div>
        <div className="grid w-24 gap-1.5"><Label htmlFor="capture-limit">Limit</Label><Input id="capture-limit" type="number" value={limit} onChange={(e) => setLimit(Number(e.target.value))} /></div>
        <Button onClick={() => load(false)} disabled={loading}>Search</Button>
      </CardContent></Card>
      <Table>
        <TableHeader><TableRow><TableHead>Time</TableHead><TableHead>User</TableHead><TableHead>Model</TableHead><TableHead>API</TableHead><TableHead>Stream</TableHead><TableHead>Tokens</TableHead><TableHead>Cost</TableHead><TableHead>Duration</TableHead><TableHead>Source</TableHead><TableHead>Request ID</TableHead></TableRow></TableHeader>
        <TableBody>
          {records.length === 0 ? <TableRow><TableCell colSpan={10} className="text-center text-muted-foreground">No captured records in this range.</TableCell></TableRow> : records.map((r) => (
            <TableRow key={`${r.request_id}-${r.ts_start}`} className="cursor-pointer" onClick={() => openCapture({ name: 'detail', requestId: r.request_id })}>
              <TableCell className="font-mono" title={r.ts_start}>{compactTime(r.ts_start)}</TableCell>
              <TableCell title={r.user_id}>{userLabel(users.find((user) => user.id === r.user_id), r.user_id)}</TableCell>
              <TableCell title={r.model_resolved ?? r.model_requested}>{r.model_resolved ?? r.model_requested}</TableCell>
              <TableCell>{r.api}</TableCell><TableCell>{r.stream ? 'yes' : 'no'}</TableCell>
              <TableCell>{r.usage ? `${r.usage.prompt_tokens}+${r.usage.completion_tokens}` : '—'}</TableCell>
              <TableCell>{r.cost_usd != null ? `$${r.cost_usd.toFixed(4)}` : '—'}</TableCell><TableCell>{r.duration_ms}ms</TableCell><TableCell>{r.source}</TableCell>
              <TableCell className="font-mono" title={r.request_id}>{r.error ? <span title={r.error}>⚠ </span> : null}{shortId(r.request_id)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {nextCursor && <Button className="w-fit" variant="outline" onClick={loadMore} disabled={loading}>{loading ? 'Loading…' : 'Load more'}</Button>}
    </div>
  )
}
