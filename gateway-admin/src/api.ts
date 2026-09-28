import type {
  CaptureRecordDetail,
  CaptureRecordsList,
  CaptureStatus,
  CreateKeyResponse,
  GatewayBudget,
  GatewayCatalog,
  GatewayKey,
  GatewayProviderStatus,
  GatewayUsage,
  GatewayUser,
  GatewayHealth,
  UsageSummary,
  BudgetResetLog,
} from './types'

const KEY_STORAGE = 'superglue.gateway.admin.key'

function query(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export function getMasterKey(): string {
  return sessionStorage.getItem(KEY_STORAGE) ?? ''
}

export function setMasterKey(key: string): void {
  sessionStorage.setItem(KEY_STORAGE, key)
}

export function clearMasterKey(): void {
  sessionStorage.removeItem(KEY_STORAGE)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  headers.set('Accept', 'application/json')
  headers.set('X-Superglue-Key', `Bearer ${getMasterKey()}`)
  if (init.body) headers.set('Content-Type', 'application/json')
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json() as { error?: { message?: string } | string }
      if (typeof body.error === 'string') message = body.error
      else if (body.error?.message) message = body.error.message
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: async (): Promise<GatewayHealth> => {
    const [liveResponse, readyResponse] = await Promise.all([fetch('/health'), fetch('/health/ready')])
    return {
      configured: true,
      gateway_url: window.location.origin,
      live: liveResponse.ok,
      ready: readyResponse.ok,
      error: null,
    }
  },
  models: () => request<{ data: { id: string }[] }>('/v1/models'),
  users: () => request<{ users: GatewayUser[] }>('/v1/users'),
  createUser: (body: { user_id: string; alias?: string; budget_id?: string }) =>
    request<GatewayUser>('/v1/users', { method: 'POST', body: JSON.stringify(body) }),
  updateUser: (id: string, body: { alias?: string; budget_id?: string | null }) =>
    request<GatewayUser>(`/v1/users/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteUser: (id: string) =>
    request<{ deleted: string; keys_deleted: number }>(`/v1/users/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  keys: () => request<{ keys: GatewayKey[] }>('/v1/keys'),
  createKey: (body: {
    user_id: string
    allowed_models: string[]
    name?: string
    expires_at?: string
  }) => request<CreateKeyResponse>('/v1/keys', { method: 'POST', body: JSON.stringify(body) }),
  updateKey: (
    id: string,
    body: {
      active?: boolean
      allowed_models?: string[]
      expires_at?: string | null
      metadata?: Record<string, unknown>
    },
  ) => request<GatewayKey>(`/v1/keys/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteKey: (id: string) => request<void>(`/v1/keys/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  budgets: () => request<{ budgets: GatewayBudget[] }>('/v1/budgets'),
  createBudget: (body: { max_budget: number; duration_sec: number; enforce: boolean }) =>
    request<GatewayBudget>('/v1/budgets', { method: 'POST', body: JSON.stringify(body) }),
  updateBudget: (id: string, body: { max_budget?: number; duration_sec?: number; enforce?: boolean }) =>
    request<GatewayBudget>(`/v1/budgets/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteBudget: (id: string) =>
    request<{ deleted: string; users_cleared: number }>(`/v1/budgets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  usage: (params: { user_id?: string; key_id?: string; limit?: number }) =>
    request<{ usage: GatewayUsage[] }>(`/v1/usage${query(params)}`),
  usageSummary: (params: {
    user_id?: string
    key_id?: string
    from?: string
    to?: string
    group_by?: string
  }) => request<UsageSummary>(`/v1/usage/summary${query(params)}`),
  budgetResets: (params: { user_id?: string; limit?: number }) =>
    request<{ resets: BudgetResetLog[] }>(`/v1/budget-resets${query(params)}`),
  providers: () => request<{ providers: GatewayProviderStatus[] }>('/v1/providers'),
  captureStatus: () => request<CaptureStatus>('/v1/capture/status'),
  captureRecords: (params: {
    user_id?: string
    request_id?: string
    model?: string
    from?: string
    to?: string
    limit?: number
    cursor?: string
  }) => request<CaptureRecordsList>(`/v1/capture/records${query(params)}`),
  captureRecord: (requestId: string, params?: { from?: string; to?: string }) =>
    request<CaptureRecordDetail>(`/v1/capture/records/${encodeURIComponent(requestId)}${query(params ?? {})}`),
}

export type { GatewayCatalog }
