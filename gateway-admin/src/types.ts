export interface AdminMeta {
  server_url: string
  gateway_configured: boolean
  gateway_url: string | null
}

export interface GatewayKeyMeta {
  gateway_user_id: string
  key_id: string
  key_prefix: string
}

export interface AdminUser {
  id: string
  username: string
  disabled: boolean
  is_admin: boolean
  created_at: string
  allowed_models: string[]
  use_server_default: boolean
  model_profile_id: string | null
  model_profile_name: string | null
  effective_models: string[]
  access_profile_id: string | null
  access_profile_name: string | null
  effective_access_profile_name: string
  gateway_key: GatewayKeyMeta | null
  has_password: boolean
}

export interface AccessProfile {
  id: string
  name: string
  description: string
  enabled: boolean
  builtin: boolean
  files: boolean
  terminal: boolean
  diff: boolean
  codesearch: boolean
  lsp: boolean
  created_at: string
  user_count: number
}

export interface ModelProfile {
  id: string
  name: string
  description: string
  allowed_models: string[]
  enabled: boolean
  budget_id: string | null
  max_reasoning_effort: string | null
  created_at: string
  user_count: number
}

export interface GatewayCatalog {
  configured: boolean
  providers: { id: string; label: string; models: string[] }[]
  defaults: string[]
  error?: string | null
}

export interface AdminStatusSnapshot {
  generated_at: string
  users: AdminUserStatus[]
  other_sessions: AdminSessionRow[]
}

export interface AdminUserStatus {
  username: string
  user_id: string
  disabled: boolean
  tunnel: 'connected' | 'disconnected'
  forward_port: number | null
  workspace_display: string | null
  local_os_user: string | null
  mount: 'none' | 'live' | 'stale'
  sessions: AdminSessionRow[]
}

export interface AdminSessionRow {
  id: string
  kind: string
  busy: boolean
  persistent: boolean
  in_memory: boolean
  workspace_display: string | null
}

export interface ClientNotification {
  id: string
  level: string
  title: string
  body: string
  expires_at: string | null
  created_at: string
}

export interface ClientSettingsBody {
  global_force_update: boolean
  maintenance_enabled: boolean
  maintenance_message: string
  maintenance_until: string | null
}

export interface NotificationsPayload {
  notifications: ClientNotification[]
  client_settings: ClientSettingsBody
  staged_releases: {
    os: string
    arch: string
    label: string
    version: string | null
    sha256_prefix: string | null
  }[]
}

export interface GatewayHealth {
  configured: boolean
  gateway_url: string | null
  live: boolean
  ready: boolean
  error?: string | null
}

export interface GatewayUser {
  id: string
  alias: string | null
  budget_id: string | null
  spend: number
  next_budget_reset_at: string | null
  created_at: string
}

export interface GatewayKey {
  id: string
  key_prefix: string
  name: string | null
  user_id: string
  active: boolean
  expires_at: string | null
  metadata_json: string | null
  allowed_models: string[]
  created_at: string
}

export interface GatewayBudget {
  id: string
  max_budget: number
  duration_sec: number
  enforce: boolean
  created_at: string
}

export interface GatewayUsage {
  id: string
  key_id: string | null
  user_id: string
  model: string
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  request_id: string | null
  created_at: string
}

export interface CreateKeyResponse {
  id: string
  key: string
  key_prefix: string
  user_id: string
  allowed_models: string[]
}

export interface UsageSummaryTotals {
  requests: number
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
}

export interface UsageSummaryRow {
  key: string
  requests: number
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
}

export interface UsageSummary {
  totals: UsageSummaryTotals
  groups: UsageSummaryRow[]
}

export interface BudgetResetLog {
  id: string
  user_id: string
  budget_id: string
  previous_spend: number
  reset_at: string
}

export interface GatewayProviderStatus {
  id: string
  label: string
  configured: boolean
  base_url: string
  key_suffix: string | null
  catalog_ok: boolean
  model_count: number
  error?: string | null
}

export interface CaptureConfigSummary {
  s3_bucket: string
  s3_prefix: string
  spool_dir: string
  rotate_bytes: number
  rotate_secs: number
  max_response_bytes: number
  exclude_users: string[]
  aws_region: string | null
  s3_endpoint: string | null
}

export interface CaptureStats {
  dropped_records: number
  pending_spool_files: number
  pending_spool_bytes: number
  channel_capacity: number
}

export interface CaptureStatus {
  enabled: boolean
  config?: CaptureConfigSummary
  stats?: CaptureStats
}

export interface CaptureRecordSummary {
  request_id: string
  ts_start: string
  duration_ms: number
  user_id: string
  key_id?: string | null
  api: 'responses' | 'chat_completions'
  model_requested: string
  model_resolved?: string | null
  stream: boolean
  usage?: { prompt_tokens: number; completion_tokens: number } | null
  cost_usd?: number | null
  error?: string | null
  sse_truncated: boolean
  source: 'spool' | 's3'
}

export interface CaptureRecordsList {
  records: CaptureRecordSummary[]
  truncated: boolean
  next_cursor?: string | null
}

export interface CaptureRecordBody {
  request_id: string
  ts_start: string
  duration_ms: number
  user_id: string
  key_id?: string | null
  api: string
  model_requested: string
  model_resolved?: string | null
  stream: boolean
  request: unknown
  response?: unknown
  sse: string[]
  sse_truncated: boolean
  usage?: { prompt_tokens: number; completion_tokens: number } | null
  cost_usd?: number | null
  error?: string | null
}

export interface CaptureRecordDetail {
  record: CaptureRecordBody
  source: 'spool' | 's3'
  source_path?: string | null
}
