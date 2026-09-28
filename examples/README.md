# superglue Rust examples

Numbered examples mirror [`../../superglue-py/examples/`](../../superglue-py/examples/) and [`../../superglue-js/examples/`](../../superglue-js/examples/).

## Prerequisites

```bash
# From repo root — loads OPENAI_API_KEY and OPENAI_MODEL
source .env

cd projects/superglue
cargo run --example 01_simple_completion
```

Set `OPENAI_BASE_URL` for Ollama or other OpenAI-compatible hosts (`08_custom_base_url`).

## Parity table

| # | Topic | Rust example |
|---|--------|----------------|
| 01 | simple completion | `01_simple_completion` |
| 02 | system prompt | `02_system_prompt` |
| 03 | single tool | `03_single_tool` |
| 04 | multiple tools | `04_multiple_tools` |
| 05 | usage tracking | `05_usage_tracking` |
| 06 | error handling | `06_error_handling` |
| 07 | max tool rounds | `07_max_tool_rounds` |
| 08 | custom base URL | `08_custom_base_url` |
| 09 | concurrent completions | `09_concurrent_completions` |
| 10 | structured data tool | `10_structured_data_tool` |
| 11 | streaming | `11_streaming` |
| 12 | streaming accumulate | `12_streaming_accumulate` |
| 13 | concurrent streams | `13_streaming_multiple_requests` |
| 14 | streaming vs buffered | `14_streaming_vs_buffered` |
| 15 | rate limiting | `15_rate_limiting` |
| 16 | retry configuration | `16_retry_configuration` |
| 17 | batch completions | `17_batch_completions` |
| 18 | hooks | `18_hooks` |
| 19 | guardrails | `19_guardrails` |
| 20 | agents | `20_agents` |
| 21 | executors | **Stub** — see Python |
| 22 | status events | `22_status_events` |
| 23 | reasoning effort | `23_reasoning_effort` |
| 24 | connection pool | `24_connection_pool` |
| 25 | model fallback | `25_model_fallback` |
| 26 | stream_response (Responses API) | `26_stream_response` |
| 27 | response threading + tools | `27_response_threading` |
| 28 | MCP tools (optional `MCP_RUN=1`) | `28_mcp_tools` |
| 29 | multi-provider `provider:model` | `29_multi_provider` |
| 30 | streaming with tool rounds | `30_streaming_tools` |
| 31 | file upload / inline attachment | `31_file_upload` |
| 32 | context optimization (GlueLLM) | `32_context_optimization` |
| 33 | context optimization benchmark (wiremock) | `33_context_optimization_benchmark` |
| 34 | context optimization benchmark (live API) | `34_context_optimization_live_benchmark` |
| 35 | programmatic tool calling (`tool_mode=Code`) | `35_code_tool_mode` |
| 37 | TypeSafe System One | `37_systemone` |
| demo | quick demo | `demo` |

Run with: `cargo run --example <name>` (no `.rs` suffix).

## Rust `Client`

Examples use [`superglue::Client`](../../src/client/mod.rs) — a high-level wrapper over `HttpClient`, `ChatOptions`, tool/hook/guardrail registries, and `complete` / `stream` / `batch` / `run_agent`. Shared env helpers live in `examples/support/mod.rs`.

## Example 21 (executors)

Executor types (`SimpleExecutor`, `AgentExecutor`, `AgentStructuredExecutor`) are Python-only. The Rust example prints a pointer to [`21_executors.py`](../../superglue-py/examples/21_executors.py).

## Batch run

From repo root:

```bash
./scripts/run-all-superglue-examples.sh
```
