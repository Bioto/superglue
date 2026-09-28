import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { GatewayKey, GatewayUser } from '../types'
import { keyLabel, userLabel } from './compactDisplay'

const ALL = 'all'

function sortedUsers(users: GatewayUser[]): GatewayUser[] {
  return [...users].sort((a, b) => userLabel(a).localeCompare(userLabel(b)))
}

export function UserSelect({
  id,
  label = 'User',
  users,
  value,
  onChange,
  allowAll = false,
  className = 'grid min-w-48 flex-1 gap-1.5',
}: {
  id?: string
  label?: string
  users: GatewayUser[]
  value: string
  onChange: (userId: string) => void
  allowAll?: boolean
  className?: string
}) {
  return (
    <div className={className}>
      {label ? <Label htmlFor={id}>{label}</Label> : null}
      <Select
        value={value || (allowAll ? ALL : undefined)}
        onValueChange={(next) => onChange(next === ALL ? '' : next)}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue placeholder={allowAll ? 'All users' : 'Select user'} />
        </SelectTrigger>
        <SelectContent>
          {allowAll ? <SelectItem value={ALL}>All users</SelectItem> : null}
          {sortedUsers(users).map((user) => (
            <SelectItem key={user.id} value={user.id}>{userLabel(user)}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export function KeySelect({
  id,
  label = 'Key',
  keys,
  users,
  userId,
  value,
  onChange,
  allowAll = true,
  className = 'grid min-w-48 flex-1 gap-1.5',
}: {
  id?: string
  label?: string
  keys: GatewayKey[]
  users: GatewayUser[]
  userId?: string
  value: string
  onChange: (keyId: string) => void
  allowAll?: boolean
  className?: string
}) {
  const userById = Object.fromEntries(users.map((user) => [user.id, user]))
  const options = keys
    .filter((key) => !userId || key.user_id === userId)
    .sort((a, b) => keyLabel(a).localeCompare(keyLabel(b)))

  return (
    <div className={className}>
      {label ? <Label htmlFor={id}>{label}</Label> : null}
      <Select
        value={value || (allowAll ? ALL : undefined)}
        onValueChange={(next) => onChange(next === ALL ? '' : next)}
      >
        <SelectTrigger id={id} className="w-full">
          <SelectValue placeholder={allowAll ? 'All keys' : 'Select key'} />
        </SelectTrigger>
        <SelectContent>
          {allowAll ? <SelectItem value={ALL}>All keys</SelectItem> : null}
          {options.map((key) => (
            <SelectItem key={key.id} value={key.id}>
              {userId ? keyLabel(key) : `${keyLabel(key)} (${userLabel(userById[key.user_id], key.user_id)})`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
