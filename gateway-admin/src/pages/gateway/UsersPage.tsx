import { useEffect, useState } from 'react'
import { api } from '../../api'
import { shortId } from '../../components/compactDisplay'
import { PaginatedTable } from '../../components/PaginatedTable'
import type { GatewayBudget, GatewayUser } from '../../types'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

function formatReset(iso: string | null) {
  if (!iso) return '—'
  const d = new Date(iso)
  const diff = d.getTime() - Date.now()
  if (diff <= 0) return 'due now'
  const hrs = Math.floor(diff / 3_600_000)
  if (hrs < 48) return `in ${hrs}h`
  return d.toLocaleDateString()
}

function SpendBar({ spend, budget }: { spend: number; budget: GatewayBudget | undefined }) {
  if (!budget) return null
  const pct = Math.min(100, (spend / budget.max_budget) * 100)
  return (
    <div className="progress-wrap" title={`$${spend.toFixed(2)} / $${budget.max_budget.toFixed(2)}`}>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
      <span className="toolbar-meta">{pct.toFixed(0)}%</span>
    </div>
  )
}

export function GatewayUsersPage() {
  const [users, setUsers] = useState<GatewayUser[]>([])
  const [budgets, setBudgets] = useState<GatewayBudget[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [form, setForm] = useState({ user_id: '', alias: '', budget_id: '' })
  const [editingAlias, setEditingAlias] = useState<{ id: string; alias: string } | null>(null)

  const budgetMap = Object.fromEntries(budgets.map((b) => [b.id, b]))

  const load = () => {
    Promise.all([api.users(), api.budgets()])
      .then(([u, b]) => {
        setUsers(u.users)
        setBudgets(b.budgets)
      })
      .catch((e) => setError(String(e)))
  }

  useEffect(() => { load() }, [])

  const create = async () => {
    try {
      await api.createUser({
        user_id: form.user_id,
        alias: form.alias || undefined,
        budget_id: form.budget_id || undefined,
      })
      setShowCreate(false)
      setForm({ user_id: '', alias: '', budget_id: '' })
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const updateBudget = async (id: string, budget_id: string) => {
    try {
      await api.updateUser(id, { budget_id: budget_id || null })
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const saveAlias = async () => {
    if (!editingAlias) return
    try {
      await api.updateUser(editingAlias.id, { alias: editingAlias.alias || undefined })
      setEditingAlias(null)
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  const remove = async (id: string) => {
    if (!confirm(`Delete gateway user ${id}?`)) return
    try {
      const result = await api.deleteUser(id)
      alert(`Deleted user; ${result.keys_deleted} key(s) removed`)
      load()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold">Gateway users</h2>
        <Button onClick={() => setShowCreate(true)}>Create user</Button>
      </div>
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      <PaginatedTable
        rows={users}
        searchKeys={(u) => [u.id, u.alias ?? '']}
        searchPlaceholder="Search users…"
      >
        {(pageRows) => (
          <Table>
              <TableHeader>
                <TableRow><TableHead>ID</TableHead><TableHead>Alias</TableHead><TableHead>Budget</TableHead><TableHead>Spend</TableHead><TableHead>Reset</TableHead><TableHead>Actions</TableHead></TableRow>
              </TableHeader>
              <TableBody>
                {pageRows.map((u) => (
                  <TableRow key={u.id}>
                    <TableCell className="font-mono" title={u.id}>{shortId(u.id)}</TableCell>
                    <TableCell>
                      <span className="cell-inline">
                        <span>{u.alias ?? '—'}</span>
                        <Button type="button" variant="outline" size="sm" onClick={() => setEditingAlias({ id: u.id, alias: u.alias ?? '' })}>Edit</Button>
                      </span>
                    </TableCell>
                    <TableCell>
                      <Select value={u.budget_id ?? 'none'} onValueChange={(value) => updateBudget(u.id, value === 'none' ? '' : value)}>
                        <SelectTrigger size="sm"><SelectValue placeholder="—" /></SelectTrigger>
                        <SelectContent><SelectItem value="none">—</SelectItem>
                        {budgets.map((b) => (
                          <SelectItem key={b.id} value={b.id}>{shortId(b.id)} (${b.max_budget})</SelectItem>
                        ))}
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell>
                      <span className="spend-cell">
                        <span className="font-mono">${u.spend.toFixed(2)}</span>
                        <SpendBar spend={u.spend} budget={u.budget_id ? budgetMap[u.budget_id] : undefined} />
                      </span>
                    </TableCell>
                    <TableCell className="font-mono">{formatReset(u.next_budget_reset_at)}</TableCell>
                    <TableCell className="actions">
                      <Button type="button" variant="destructive" size="sm" onClick={() => remove(u.id)}>Delete</Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
        )}
      </PaginatedTable>
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent>
          <DialogHeader><DialogTitle>Create gateway user</DialogTitle></DialogHeader>
          <div className="grid gap-3">
            <div className="grid gap-1.5"><Label htmlFor="gateway-user-id">User ID</Label><Input id="gateway-user-id" value={form.user_id} onChange={(e) => setForm({ ...form, user_id: e.target.value })} /></div>
            <div className="grid gap-1.5"><Label htmlFor="gateway-user-alias">Alias</Label><Input id="gateway-user-alias" value={form.alias} onChange={(e) => setForm({ ...form, alias: e.target.value })} /></div>
            <div className="grid gap-1.5"><Label>Budget</Label>
              <Select value={form.budget_id || 'none'} onValueChange={(value) => setForm({ ...form, budget_id: value === 'none' ? '' : value })}>
                <SelectTrigger><SelectValue placeholder="—" /></SelectTrigger>
                <SelectContent><SelectItem value="none">—</SelectItem>{budgets.map((b) => <SelectItem key={b.id} value={b.id}>{b.id.slice(0, 8)}…</SelectItem>)}</SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter><Button type="button" variant="outline" onClick={() => setShowCreate(false)}>Cancel</Button><Button disabled={!form.user_id} onClick={create}>Create</Button></DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={editingAlias !== null} onOpenChange={(open) => { if (!open) setEditingAlias(null) }}>
        <DialogContent>
          <DialogHeader><DialogTitle>Edit alias</DialogTitle></DialogHeader>
          {editingAlias && <div className="grid gap-1.5"><Label htmlFor="gateway-user-edit-alias">Alias</Label><Input id="gateway-user-edit-alias" value={editingAlias.alias} onChange={(e) => setEditingAlias({ ...editingAlias, alias: e.target.value })} autoFocus /></div>}
          <DialogFooter><Button type="button" variant="outline" onClick={() => setEditingAlias(null)}>Cancel</Button><Button onClick={saveAlias}>Save</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
