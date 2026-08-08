# Multi-provider routing and file attachments

Superglue chat models use a `provider:model` string (for example `openai:gpt-4o-mini` or `anthropic:claude-sonnet-4-20250514`). The provider prefix selects which API credentials and base URL are used; the model suffix is sent to that provider's API.

## Environment variables

| Provider | API key env | Base URL env (optional) |
|----------|-------------|-------------------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| xAI | `XAI_API_KEY` | `XAI_BASE_URL` |
| Groq | `GROQ_API_KEY` | `GROQ_BASE_URL` |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |

Bindings load these via `ProviderCredentials::from_env()` when constructing a client. You can override per provider with `api_keys` (Python/JS/Kotlin) or `ClientBuilder` in Rust.

## Per-key rate limits

Pass `requests_per_second_for` (Python) / `requestsPerSecondFor` (JS/Kotlin) as a map from provider name to QPS. The legacy single `requests_per_second` / `quota_per_second` applies to OpenAI when no per-provider map is set.

## File uploads

Two patterns:

1. **Inline bytes** — build a user message with `message_with_file_bytes` / `messageWithFileBytes` / `messageWithFileBytes` (Kotlin) and pass it to `complete_messages`.
2. **Provider file API** — `upload_file` posts to the provider's files endpoint and returns a `file_id` for use in chat messages.

Set `max_upload_bytes` to cap inline attachment size (default 20 MB).

## Streaming with tools

When tools are registered on the client, `stream()` runs the same multi-round tool loop as `complete()`, emitting token deltas between rounds. Kotlin, Python, and JS bindings all use `stream_complete_with_tools` when the tool registry is non-empty.

## Examples

- Rust: `cargo run --example 29_multi_provider` (from `projects/superglue/`)
- Python: `python examples/29_multi_provider.py`
- JS: `node examples/29_multi_provider.mjs`
- Kotlin: `FeatureExamplesTest.example29_multi_provider`
