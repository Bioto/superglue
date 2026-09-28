import type { GatewayKey, GatewayUser } from '../types'

export function shortId(value: string | null | undefined): string {
  if (!value) return '—'
  return value.length > 12 ? `${value.slice(0, 8)}…` : value
}

export function userLabel(user: GatewayUser | undefined, fallbackId?: string | null): string {
  const alias = user?.alias?.trim()
  if (alias) return alias
  return shortId(user?.id ?? fallbackId)
}

export function keyLabel(key: GatewayKey | undefined, fallbackId?: string | null): string {
  const name = key?.name?.trim()
  if (name) return name
  if (key?.key_prefix) return shortId(key.key_prefix)
  return shortId(key?.id ?? fallbackId)
}

export function compactTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
