# LLM Gateway

The superglue **LLM gateway** is an optional HTTP server (`gateway` feature) that sits between your applications and upstream LLM providers. It exposes an OpenAI-compatible API with governance features inspired by [any-llm-gateway](https://github.com/mozilla-ai/any-llm):

- **Virtual API keys** with per-key model allowlists (stored in SQLite)
- **Master key** for admin operations
- **Users and budgets** with lazy auto-reset
- **Usage analytics** (tokens, estimated cost, attribution)

## Quick start

Build and run with the gateway feature:

```bash
cargo build --features gateway --release

export GATEWAY_MASTER_KEY="your-secure-master-key"
export OPENAI_API_KEY="sk-..."

superglue gateway serve \
  --addr 0.0.0.0:8080 \
  --db ./superglue-gateway.db
```

## CLI management

Manage the gateway via a local SQLite file or a remote HTTPS admin API.

**Local** (default): reads/writes `--db` (default `./superglue-gateway.db`).

**Remote**: pass `--url` and `--master-key` (or env `SUPERGLUE_GATEWAY_URL` / `GATEWAY_MASTER_KEY`):

```bash
export SUPERGLUE_GATEWAY_URL=https://gateway.myharn.sh
export GATEWAY_MASTER_KEY=...

superglue gateway --url "$SUPERGLUE_GATEWAY_URL" user list
superglue gateway --url "$SUPERGLUE_GATEWAY_URL" key create --user-id alice --model 'openai:*'
```

All manage commands accept `--output pretty|json`. Remote mode uses the same subcommands as local mode.

```bash
# Users
superglue gateway user create --user-id alice --alias Alice --budget-id <budget-id>
superglue gateway user list
superglue gateway user update --user-id alice --budget-id <new-budget-id>

# Virtual keys (model flag is repeatable)
superglue gateway key create --user-id alice --model openai:gpt-4o-mini --model anthropic:*
superglue gateway key list
superglue gateway key update --id <key-id> --active false
superglue gateway key delete --id <key-id>

# Budgets
superglue gateway budget create --max-budget 10 --duration-sec 2592000
superglue gateway budget create --max-budget 50 --duration-sec 2592000 --enforce false  # track-only
superglue gateway budget list

# Usage
superglue gateway usage list --user-id alice --limit 50
superglue gateway usage list --output json

# Models (allowed patterns for a key; master key shows unrestricted *)
superglue gateway model list
superglue gateway model list --key sgw-...
superglue gateway --url "$SUPERGLUE_GATEWAY_URL" model list
```

Key creation prints the plaintext secret **once** — store it immediately.

## Authentication

Clients authenticate with either header (`X-Superglue-Key` takes precedence):

```
Authorization: Bearer <key>
X-Superglue-Key: Bearer <key>
```

- **Master key** — full admin access; completion requests must include `"user": "<user_id>"` in the JSON body
- **Virtual keys** — scoped to allowed models; usage is attributed to the key's linked user automatically

## Admin setup (HTTP API)

Alternatively, use the HTTP admin API while the server is running:

```bash
# Create a budget ($10/month, enforced)
curl -X POST http://localhost:8080/v1/budgets \
  -H "X-Superglue-Key: Bearer $GATEWAY_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"max_budget": 10.0, "duration_sec": 2592000, "enforce": true}'

# Create a user with that budget
curl -X POST http://localhost:8080/v1/users \
  -H "X-Superglue-Key: Bearer $GATEWAY_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "alias": "Alice", "budget_id": "<budget-id>"}'

# Create a virtual key with model allowlist
curl -X POST http://localhost:8080/v1/keys \
  -H "X-Superglue-Key: Bearer $GATEWAY_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "alice-app",
    "user_id": "alice",
    "allowed_models": ["openai:gpt-4o-mini", "anthropic:*"]
  }'
```

The plaintext key is returned **once** in the response. Store it securely.

## Model allowlist

Each virtual key requires at least one pattern in `allowed_models`:

| Pattern | Matches |
|---------|---------|
| `openai:gpt-4o-mini` | Exact model string |
| `openai:*` | Any OpenAI model |
| `*` | All models |

Keys with no allowlist rows are denied (explicit opt-in). The master key is unrestricted.

`GET /v1/models` returns allowed model patterns for scoped keys. For the master key (or `*` allowlist), it fetches and merges model ids from each configured upstream provider (`openai:…`, `anthropic:…`, etc.) via each provider's `/v1/models` API.

## Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "X-Superglue-Key: Bearer sgw-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Models use the `provider:model` format (`openai:gpt-4o-mini`, `anthropic:claude-3-5-sonnet-20241022`, etc.).

Streaming is supported via `"stream": true`.

## Responses API

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "X-Superglue-Key: Bearer sgw-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "input": "Hello!",
    "stream": false
  }'
```

Supports `openai`, `xai`, and `groq` via OpenAI-compatible `/v1/responses`, and `anthropic` via Messages translation. Streaming uses `"stream": true` (SSE).

## Budget enforcement

- **enforce=true** — reject requests when `spend >= max_budget` (HTTP 429)
- **enforce=false** — track spend but never reject (track-only mode)
- Budgets reset lazily per user when `next_budget_reset_at` passes

**v1 limitation:** there is no cost reservation. A single large request may push spend above the budget limit after the pre-check passes.

## Costing

Token counts are logged from provider responses. USD cost uses the static rates in [`src/costing/mod.rs`](../src/costing/mod.rs). Unknown models (most Anthropic/xAI/Groq models today) log `$0.00` cost — expand the pricing table for accurate analytics.

## Provider credentials

Upstream provider API keys are loaded from environment variables on the server (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_BASE_URL`). Clients never see provider credentials.

## API reference

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | none | Liveness |
| GET | `/health/ready` | none | Readiness (DB ping) |
| POST | `/v1/chat/completions` | any key | Proxy completion |
| POST | `/v1/responses` | any key | Proxy Responses API (OpenAI-shaped; multi-provider) |
| GET | `/v1/models` | any key | List allowed models |
| POST/GET | `/v1/keys` | master | Create/list virtual keys |
| PATCH/DELETE | `/v1/keys/{id}` | master | Update/revoke keys |
| POST/GET | `/v1/users` | master | Create/list users |
| PATCH | `/v1/users/{id}` | master | Update user |
| POST/GET | `/v1/budgets` | master | Create/list budgets |
| GET | `/v1/usage` | master | Usage logs |

## SQLite

The gateway uses a single SQLite file (WAL mode) for keys, users, budgets, and usage logs. Default path: `./superglue-gateway.db`.

Virtual key secrets are stored as SHA-256 hashes only. Key verification uses constant-time comparison.
