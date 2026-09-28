import { useEffect, useState } from 'react'
import { api } from '../../api'
import type { GatewayProviderStatus } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

export function GatewayProvidersPage() {
  const [providers, setProviders] = useState<GatewayProviderStatus[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.providers()
      .then((r) => setProviders(r.providers))
      .catch((e) => setError(String(e)))
  }, [])

  return (
    <div className="grid gap-3">
      <Card>
        <CardHeader><CardTitle>Gateway providers</CardTitle></CardHeader>
        <CardContent>
          {error && <Alert variant="destructive" className="mb-3"><AlertDescription>{error}</AlertDescription></Alert>}
          <Table>
            <TableHeader><TableRow><TableHead>Provider</TableHead><TableHead>Configured</TableHead><TableHead>Base URL</TableHead><TableHead>Key</TableHead><TableHead>Catalog</TableHead><TableHead>Models</TableHead></TableRow></TableHeader>
            <TableBody>
              {providers.map((p) => (
                <TableRow key={p.id}>
                  <TableCell className="font-medium">{p.label}</TableCell>
                  <TableCell><Badge variant={p.configured ? 'default' : 'secondary'}>{p.configured ? 'yes' : 'no'}</Badge></TableCell>
                  <TableCell className="font-mono" title={p.base_url}>{p.base_url}</TableCell>
                  <TableCell className="font-mono">{p.key_suffix ?? '—'}</TableCell>
                  <TableCell>{p.configured ? <Badge variant={p.catalog_ok ? 'default' : 'destructive'}>{p.catalog_ok ? 'ok' : 'error'}</Badge> : '—'}</TableCell>
                  <TableCell>{p.catalog_ok ? p.model_count : (p.error ?? '—')}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
