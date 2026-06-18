# superglue-js examples

Runnable scripts mirror [`../../superglue-py/examples/`](../../superglue-py/examples/) by number and topic. Build first, then run from the `superglue-js/` package root:

```bash
npm install
npm run build
export OPENAI_API_KEY=sk-...
node examples/01_simple_completion.mjs
```

Use `OPENAI_MODEL` and `OPENAI_BASE_URL` where noted (e.g. `08_custom_base_url.mjs`).

Rust equivalents: [`../../superglue/examples/`](../../superglue/examples/) — `cargo run --example 01_simple_completion` from `projects/superglue/`.

## Parity table

| # | Topic | Python | JS | Rust |
|---|--------|--------|-----|------|
| 01 | simple completion | `01_simple_completion.py` | `01_simple_completion.mjs` | `01_simple_completion` |
| 02 | system prompt | `02_system_prompt.py` | `02_system_prompt.mjs` | `02_system_prompt` |
| 03 | single tool | `03_single_tool.py` | `03_single_tool.mjs` | `03_single_tool` |
| 04 | multiple tools | `04_multiple_tools.py` | `04_multiple_tools.mjs` | `04_multiple_tools` |
| 05 | usage tracking | `05_usage_tracking.py` | `05_usage_tracking.mjs` | `05_usage_tracking` |
| 06 | error handling | `06_error_handling.py` | `06_error_handling.mjs` | `06_error_handling` |
| 07 | max tool rounds | `07_max_tool_rounds.py` | `07_max_tool_rounds.mjs` | `07_max_tool_rounds` |
| 08 | custom base URL | `08_custom_base_url.py` | `08_custom_base_url.mjs` | `08_custom_base_url` |
| 09 | concurrent completions | `09_concurrent_completions.py` | `09_concurrent_completions.mjs` | `09_concurrent_completions` |
| 10 | structured data tool | `10_structured_data_tool.py` | `10_structured_data_tool.mjs` | `10_structured_data_tool` |
| 11 | streaming | `11_streaming.py` | `11_streaming.mjs` | `11_streaming` |
| 12 | streaming accumulate | `12_streaming_accumulate.py` | `12_streaming_accumulate.mjs` | `12_streaming_accumulate` |
| 13 | concurrent streams | `13_streaming_multiple_requests.py` | `13_streaming_multiple_requests.mjs` | `13_streaming_multiple_requests` |
| 14 | streaming vs buffered | `14_streaming_vs_buffered.py` | `14_streaming_vs_buffered.mjs` | `14_streaming_vs_buffered` |
| 15 | rate limiting | `15_rate_limiting.py` | `15_rate_limiting.mjs` | `15_rate_limiting` |
| 16 | retry configuration | `16_retry_configuration.py` | `16_retry_configuration.mjs` | `16_retry_configuration` |
| 17 | batch completions | `17_batch_completions.py` | `17_batch_completions.mjs` | `17_batch_completions` |
| 18 | hooks | `18_hooks.py` | `18_hooks.mjs` | `18_hooks` |
| 19 | guardrails | `19_guardrails.py` | `19_guardrails.mjs` | `19_guardrails` |
| 20 | agents | `20_agents.py` | `20_agents.mjs` | `20_agents` |
| 21 | executors | `21_executors.py` | **Not in JS** — see below | **Stub** — see below |
| 22 | status events | `22_status_events.py` | `22_status_events.mjs` | `22_status_events` |
| 23 | reasoning effort | `23_reasoning_effort.py` | `23_reasoning_effort.mjs` | `23_reasoning_effort` |
| 24 | connection pool | `24_connection_pool.py` | `24_connection_pool.mjs` | `24_connection_pool` |
| 25 | model fallback | `25_model_fallback.py` | `25_model_fallback.mjs` | `25_model_fallback` |
| 26 | stream response | `26_stream_response.py` | `26_stream_response.mjs` | `26_stream_response` |
| 27 | response threading | `27_complete_response.py` | `27_complete_response.mjs` | `27_response_threading` |
| 28 | MCP tools | `28_mcp_tools.py` | `28_mcp_tools.mjs` | `28_mcp_tools` |
| 29 | multi-provider | `29_multi_provider.py` | `29_multi_provider.mjs` | `29_multi_provider` |
| 30 | streaming tools | `30_streaming_tools.py` | `30_streaming_tools.mjs` | `30_streaming_tools` |
| 31 | file upload | `31_file_upload.py` | `31_file_upload.mjs` | `31_file_upload` |
| demo | quick demo | `demo.py` | `demo.mjs` | `demo` |

## Example 21 (executors)

The Python bindings include `SimpleExecutor`, `AgentExecutor`, and `AgentStructuredExecutor` (gluellm-style executor wrappers). These types are **not** exposed in the Node.js / napi-rs binding or the native Rust crate. Use `Client`, `AgentEngine`, hooks, and guardrails directly, or run [`21_executors.py`](../../superglue-py/examples/21_executors.py) in Python. Rust runs [`21_executors`](../../superglue/examples/21_executors.rs) as an informational stub.
