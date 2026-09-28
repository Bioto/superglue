import { useEffect, useState } from 'react'
import { api } from '../../api'
import { GatewayNavButton } from '../../components/agent/gatewayNav'
import { compactTime, keyLabel, shortId, userLabel } from '../../components/compactDisplay'
import type { CaptureRecordDetail, GatewayKey, GatewayUser } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table'

function JsonBlock({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(true)
  if (value == null) return null
  return (
    <Card>
      <CardHeader className="pb-2"><Button type="button" variant="ghost" className="w-fit" onClick={() => setOpen((v) => !v)}>{open ? '▼' : '▶'} {title}</Button></CardHeader>
      {open && <CardContent><pre className="m-0 max-h-96 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">{JSON.stringify(value, null, 2)}</pre></CardContent>}
    </Card>
  )
}

export function GatewayCaptureRecordDetailPage({ requestId }: { requestId?: string } = {}) {
  const [detail, setDetail] = useState<CaptureRecordDetail | null>(null)
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [keys, setKeys] = useState<GatewayKey[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.users(), api.keys()])
      .then(([u, k]) => {
        setUsers(u.users)
        setKeys(k.keys)
      })
      .catch(() => {
        setUsers([])
        setKeys([])
      })
  }, [])

  useEffect(() => {
    if (!requestId) return
    setError(null)
    api.captureRecord(requestId).then(setDetail).catch((e) => setError(String(e)))
  }, [requestId])

  if (!requestId) return <div className="p-4 text-sm text-muted-foreground">Missing request ID.</div>
  if (error) return <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>
  if (!detail) return <div className="p-4 text-sm text-muted-foreground">Loading record…</div>

  const { record } = detail
  const user = users.find((item) => item.id === record.user_id)
  const key = record.key_id ? keys.find((item) => item.id === record.key_id) : undefined
  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3"><h2 className="text-lg font-semibold">Capture record</h2><GatewayNavButton capture={{ name: 'records' }}>Back to records</GatewayNavButton></div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        <Summary label="Request ID" value={shortId(record.request_id)} mono />
        <Summary label="User" value={userLabel(user, record.user_id)} />
        <Summary label="Model" value={record.model_resolved ?? record.model_requested} />
        <Summary label="Duration" value={`${record.duration_ms}ms`} />
        <Summary label="Cost" value={record.cost_usd != null ? `$${record.cost_usd.toFixed(4)}` : '—'} />
      </div>
      {record.error && <Alert variant="destructive"><AlertDescription>{record.error}</AlertDescription></Alert>}
      <Card><CardContent><Table><TableBody>
        <TableRow><TableCell className="font-medium">Started</TableCell><TableCell className="font-mono">{compactTime(record.ts_start)}</TableCell></TableRow>
        <TableRow><TableCell className="font-medium">API</TableCell><TableCell>{record.api}</TableCell></TableRow>
        <TableRow><TableCell className="font-medium">Stream</TableCell><TableCell>{record.stream ? 'yes' : 'no'}</TableCell></TableRow>
        <TableRow><TableCell className="font-medium">Key</TableCell><TableCell>{record.key_id ? keyLabel(key, record.key_id) : '—'}</TableCell></TableRow>
        <TableRow><TableCell className="font-medium">Source</TableCell><TableCell>{detail.source}{detail.source_path ? ` · ${detail.source_path}` : ''}</TableCell></TableRow>
        <TableRow><TableCell className="font-medium">Tokens</TableCell><TableCell>{record.usage ? `${record.usage.prompt_tokens} prompt + ${record.usage.completion_tokens} completion` : '—'}</TableCell></TableRow>
      </TableBody></Table></CardContent></Card>
      <JsonBlock title="Request" value={record.request} />
      {!record.stream && <JsonBlock title="Response" value={record.response} />}
      {record.stream && record.sse.length > 0 && (
        <Card><CardHeader><CardTitle>SSE events ({record.sse.length}){record.sse_truncated ? ' · truncated' : ''}</CardTitle></CardHeader><CardContent>
          <Table><TableBody>{record.sse.map((event, i) => <TableRow key={i}><TableCell>{i + 1}</TableCell><TableCell className="whitespace-pre-wrap font-mono text-xs">{event}</TableCell></TableRow>)}</TableBody></Table>
        </CardContent></Card>
      )}
    </div>
  )
}

function Summary({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">{label}</div><div className={mono ? 'font-mono font-semibold' : 'font-semibold'}>{value}</div></CardContent></Card>
}
