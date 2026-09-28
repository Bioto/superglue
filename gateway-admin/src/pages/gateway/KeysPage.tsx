import { useEffect, useState } from 'react'
import { api } from '../../api'
import { shortId, userLabel } from '../../components/compactDisplay'
import { ChipRow } from '../../components/compactUi'
import { UserSelect } from '../../components/GatewayEntitySelect'
import { ModelPicker } from '../../components/ModelPicker'
import { PaginatedTable } from '../../components/PaginatedTable'
import type { CreateKeyResponse, GatewayKey, GatewayUser } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

function parseMetadata(raw: string | null): Record<string, unknown> {
  if (!raw) return {}
  try {
    return JSON.parse(raw) as Record<string, unknown>
  } catch {
    return {}
  }
}

function expiresBadge(iso: string | null) {
  if (!iso) return null
  const d = new Date(iso)
  const soon = d.getTime() - Date.now() < 7 * 86400_000
  return (
    <span className={`status-badge ${soon ? 'status-disabled' : 'status-live'}`}>
      {d.toLocaleDateString()}
    </span>
  )
}

export function GatewayKeysPage() {
  const [keys, setKeys] = useState<GatewayKey[]>([])
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [editing, setEditing] = useState<GatewayKey | null>(null)
  const [plaintextKey, setPlaintextKey] = useState<CreateKeyResponse | null>(null)
  const [form, setForm] = useState({ user_id: '', name: '', models: [] as string[], useDefault: true, expires_at: '' })
  const [editForm, setEditForm] = useState({ models: [] as string[], expires_at: '' })

  const userById = Object.fromEntries(users.map((user) => [user.id, user]))

  const load = () => {
    Promise.all([api.keys(), api.users()])
      .then(([k, u]) => {
        setKeys(k.keys)
        setUsers(u.users)
      })
      .catch((e) => setError(String(e)))
  }
  useEffect(() => {
    load()
  }, [])

  const defaultModels = ['openai:*', 'anthropic:*']

  const create = async () => {
    try {
      const result = await api.createKey({
        user_id: form.user_id,
        name: form.name || undefined,
        allowed_models: form.useDefault ? defaultModels : form.models,
        expires_at: form.expires_at || undefined,
      })
      setPlaintextKey(result)
      setShowCreate(false)
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const openEdit = (key: GatewayKey) => {
    setEditing(key)
    setEditForm({ models: [...key.allowed_models], expires_at: key.expires_at?.slice(0, 16) ?? '' })
  }

  const saveEdit = async () => {
    if (!editing) return
    try {
      await api.updateKey(editing.id, {
        allowed_models: editForm.models,
        expires_at: editForm.expires_at ? new Date(editForm.expires_at).toISOString() : null,
      })
      setEditing(null)
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const toggleActive = async (key: GatewayKey) => {
    try {
      await api.updateKey(key.id, { active: !key.active })
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const remove = async (id: string) => {
    if (!confirm('Revoke this key?')) return
    try {
      await api.deleteKey(id)
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Gateway keys</h2>
        <Button onClick={() => setShowCreate(true)}>Create key</Button>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      <PaginatedTable rows={keys} searchKeys={(k) => [k.key_prefix, k.name ?? '', k.user_id, userLabel(userById[k.user_id], k.user_id)]} searchPlaceholder="Search keys…">
        {(pageRows) => (
          <Table>
              <TableHeader>
                <TableRow><TableHead>Prefix</TableHead><TableHead>Name</TableHead><TableHead>User</TableHead><TableHead>Active</TableHead><TableHead>Expires</TableHead><TableHead>Metadata</TableHead><TableHead>Models</TableHead><TableHead>Actions</TableHead></TableRow>
              </TableHeader>
              <TableBody>
                {pageRows.map((k) => {
                  const meta = parseMetadata(k.metadata_json)
                  return (
                    <TableRow key={k.id}>
                      <TableCell className="font-mono" title={k.key_prefix}>{shortId(k.key_prefix)}</TableCell>
                      <TableCell>{k.name ?? '—'}</TableCell>
                      <TableCell title={k.user_id}>{userLabel(userById[k.user_id], k.user_id)}</TableCell>
                      <TableCell><Badge variant={k.active ? 'default' : 'secondary'}>{k.active ? 'yes' : 'no'}</Badge></TableCell>
                      <TableCell>{expiresBadge(k.expires_at) ?? '—'}</TableCell>
                      <TableCell className="font-mono">
                        {meta.max_reasoning_effort ? `max: ${String(meta.max_reasoning_effort)}` : '—'}
                      </TableCell>
                      <TableCell><ChipRow items={k.allowed_models} /></TableCell>
                      <TableCell className="actions">
                        <Button type="button" variant="outline" size="sm" onClick={() => openEdit(k)}>Edit</Button>
                        <Button type="button" variant="outline" size="sm" onClick={() => toggleActive(k)}>{k.active ? 'Disable' : 'Enable'}</Button>
                        <Button type="button" variant="destructive" size="sm" onClick={() => remove(k.id)}>Revoke</Button>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
        )}
      </PaginatedTable>
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader><DialogTitle>Create gateway key</DialogTitle></DialogHeader>
          <div className="grid gap-3">
            <UserSelect id="gateway-key-user" users={users} value={form.user_id} onChange={(user_id) => setForm({ ...form, user_id })} className="grid gap-1.5" />
            <div className="grid gap-1.5"><Label htmlFor="gateway-key-name">Name</Label><Input id="gateway-key-name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></div>
            <div className="grid gap-1.5"><Label htmlFor="gateway-key-expires">Expires</Label><Input id="gateway-key-expires" type="datetime-local" value={form.expires_at} onChange={(e) => setForm({ ...form, expires_at: e.target.value })} /></div>
            <ModelPicker selected={form.models} useDefault={form.useDefault} onChange={(models, useDefault) => setForm({ ...form, models, useDefault })} />
          </div>
          <DialogFooter><Button type="button" variant="outline" onClick={() => setShowCreate(false)}>Cancel</Button><Button disabled={!form.user_id} onClick={create}>Create</Button></DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={editing !== null} onOpenChange={(open) => { if (!open) setEditing(null) }}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader><DialogTitle>Edit key {editing?.key_prefix}</DialogTitle></DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5"><Label htmlFor="gateway-key-edit-expires">Expires (clear to remove)</Label><Input id="gateway-key-edit-expires" type="datetime-local" value={editForm.expires_at} onChange={(e) => setEditForm({ ...editForm, expires_at: e.target.value })} /></div>
            <ModelPicker selected={editForm.models} useDefault={false} hideDefaultOption onChange={(models) => setEditForm({ ...editForm, models })} />
          </div>
          <DialogFooter><Button type="button" variant="outline" onClick={() => setEditing(null)}>Cancel</Button><Button disabled={editForm.models.length === 0} onClick={saveEdit}>Save</Button></DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={plaintextKey !== null} onOpenChange={(open) => { if (!open) setPlaintextKey(null) }}>
        <DialogContent>
          <DialogHeader><DialogTitle>Key created — copy now</DialogTitle><DialogDescription>This secret is shown once.</DialogDescription></DialogHeader>
          {plaintextKey && <pre className="max-h-48 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">{plaintextKey.key}</pre>}
          <DialogFooter><Button onClick={() => { if (plaintextKey) void navigator.clipboard.writeText(plaintextKey.key); setPlaintextKey(null) }}>Copy & close</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
