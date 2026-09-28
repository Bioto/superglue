import { useEffect, useState } from 'react'
import { api } from '../../api'
import { GatewayNavButton } from '../../components/agent/gatewayNav'
import { userLabel } from '../../components/compactDisplay'
import type { CaptureStatus, GatewayUser } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export function GatewayCapturePage() {
  const [status, setStatus] = useState<CaptureStatus | null>(null)
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = () => {
    setError(null)
    Promise.all([api.captureStatus(), api.users().catch(() => ({ users: [] }))])
      .then(([s, u]) => {
        setStatus(s)
        setUsers(u.users)
      })
      .catch((e) => setError(String(e)))
  }

  useEffect(() => { load() }, [])

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Gateway capture</h2>
        <GatewayNavButton capture={{ name: 'records' }}>Browse records</GatewayNavButton>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      {!status && !error && <div className="p-4 text-sm text-muted-foreground">Loading capture status…</div>}
      {status && !status.enabled && (
        <Alert><AlertDescription>Capture is disabled on the gateway. Set <code>SUPERGLUE_CAPTURE_S3_BUCKET</code> and redeploy with the <code>capture</code> feature, then restart the gateway.</AlertDescription></Alert>
      )}
      {status?.enabled && status.stats && (
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Status</div><div className="font-semibold">Enabled</div></CardContent></Card>
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Dropped records</div><div className="font-mono font-semibold">{status.stats.dropped_records.toLocaleString()}</div></CardContent></Card>
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Pending spool files</div><div className="font-mono font-semibold">{status.stats.pending_spool_files}</div></CardContent></Card>
          <Card size="sm"><CardContent><div className="text-xs text-muted-foreground">Pending spool size</div><div className="font-mono font-semibold">{formatBytes(status.stats.pending_spool_bytes)}</div></CardContent></Card>
        </div>
      )}
      {status?.stats && status.stats.dropped_records > 0 && (
        <Alert variant="destructive"><AlertDescription>{status.stats.dropped_records.toLocaleString()} records were dropped because the capture writer fell behind. Consider increasing spool throughput or reducing traffic.</AlertDescription></Alert>
      )}
      {status?.config && (
        <Card>
          <CardHeader><CardTitle>Configuration</CardTitle></CardHeader>
          <CardContent className="grid gap-3">
            <p className="m-0 text-sm text-muted-foreground">Capture settings are read from gateway environment variables. Changing them requires a gateway restart.</p>
            <dl className="grid gap-2 text-sm sm:grid-cols-2">
              <ConfigRow label="S3 bucket" value={status.config.s3_bucket} mono />
              <ConfigRow label="S3 prefix" value={status.config.s3_prefix} mono />
              <ConfigRow label="Spool directory" value={status.config.spool_dir} mono />
              <ConfigRow label="Rotate bytes" value={formatBytes(status.config.rotate_bytes)} />
              <ConfigRow label="Rotate seconds" value={`${status.config.rotate_secs}s`} />
              <ConfigRow label="Max response bytes" value={formatBytes(status.config.max_response_bytes)} />
              <ConfigRow label="Channel capacity" value={String(status.stats?.channel_capacity ?? '—')} />
              <ConfigRow label="AWS region" value={status.config.aws_region ?? '—'} />
              <ConfigRow label="S3 endpoint" value={status.config.s3_endpoint ?? 'default'} mono />
              <ConfigRow label="Excluded users" value={status.config.exclude_users.length === 0 ? '—' : status.config.exclude_users.map((id) => userLabel(users.find((user) => user.id === id), id)).join(', ')} />
            </dl>
            <Button className="w-fit" onClick={load}>Refresh</Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function ConfigRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return <div className="rounded-md border border-border p-2"><dt className="text-xs text-muted-foreground">{label}</dt><dd className={mono ? 'mt-1 break-all font-mono text-xs' : 'mt-1'}>{value}</dd></div>
}
