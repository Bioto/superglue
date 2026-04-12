# Architecture

## Summary

The **Rust core** owns network I/O to LLM providers, SSE streaming, retries and rate limiting, the tool dispatch loop, and (planned) workflow and hook orchestration. **Language bindings** register tools, supply configuration, and surface results and telemetry to the host in an idiomatic way.

## Implemented module map

```
superglue/                      — Rust library crate
├── src/
│   ├── http/
│   │   ├── client.rs           — HttpClient: GET, POST JSON, streaming POST (SSE)
│   │   ├── retry.rs            — RetryPolicy (GET: all retryable; POST: transport + 429/5xx)
│   │   ├── rate_limit.rs       — governor token-bucket rate limiter
│   │   ├── sse.rs              — SseParser: incremental SSE framing
│   │   ├── url.rs              — join_base_url helper
│   │   └── error.rs            — unified HTTP error taxonomy
│   ├── openai/
│   │   └── mod.rs              — Full OpenAI chat completion types (request + response + streaming)
│   ├── chat/
│   │   └── mod.rs              — complete_with_tools, stream_complete, ChatOptions, CompletionOutcome
│   ├── tools/
│   │   ├── registry.rs         — ToolRegistry: register + dispatch by name
│   │   ├── types.rs            — ToolSpec, ToolInvocation; From<proto::ToolSpec>
│   │   ├── harness.rs          — offline scripted tool execution plans
│   │   └── error.rs            — ToolInvokeError
│   ├── proto.rs                — includes prost-generated types from proto/superglue.proto
│   └── lib.rs                  — public re-exports + version()

superglue-py/                   — Python extension crate (PyO3, maturin)
├── src/lib.rs                  — Client, CompletionOutcome, StreamOutcome pyclass impls
└── examples/                   — 14 example scripts (01_simple_completion … 14_streaming_vs_buffered)
```

## Responsibilities

### Rust core — what is implemented

| Module | Responsibility |
|--------|---------------|
| `http::HttpClient` | `reqwest` wrapper; GET with full retries, POST JSON with POST-safe retries, streaming POST returning a byte stream |
| `http::RetryPolicy` | Configurable max retries and exponential backoff; GET retries all retryable statuses; POST retries only transport errors + 429/502/503/504 |
| `http::rate_limit` | Per-second quota via `governor`; optional, applied before every request |
| `http::SseParser` | Incremental buffer-based SSE parser; handles split chunks, comments, multi-line data |
| `openai` | Full `ChatCompletionRequest` (all OpenAI params), `ChatCompletionResponse`, `ChatMessage` with multipart content, `ChatCompletionChunk` for streaming, `Usage`, `ToolCall`, `ToolChoice`, `ResponseFormat`, `StopSequence`, etc. |
| `chat::complete_with_tools` | Multi-turn tool loop: sends POST, checks for tool calls, dispatches to `ToolRegistry`, appends results, repeats until text response or `max_tool_rounds` |
| `chat::stream_complete` | Single-turn streaming: POST with `stream:true`, parses SSE chunks, calls `on_delta(token)` for each content delta, returns `StreamOutcome` |
| `tools::ToolRegistry` | Async `Arc<dyn Tool>` store; concurrent-safe registration and dispatch by name |
| `tools::harness` | Deterministic offline test harness: execute a scripted plan of tool invocations without a live model |
| `proto` | `prost`-generated types from `proto/superglue.proto`; covers `ChatOptions`, `ChatMessage`, `Usage`, `ToolSpec`, `CompletionOutcome` |

### Language bindings — Python (`superglue-py`)

| Feature | Implementation |
|---------|---------------|
| `Client(api_key, model, …)` | `PyClient` wraps `ChatOptions` + `ToolRegistry` + `HttpClient` |
| `client.register_tool(name, description, parameters, fn)` | Wraps a Python `def fn(args: dict) -> dict` as a `PythonDictTool`; converts via `json.loads`/`json.dumps` |
| `client.complete(message)` | Calls `complete_with_tools` via `block_on`; tool callbacks delivered via `spawn_blocking + Python::attach` |
| `client.stream(message, on_token)` | Calls `stream_complete` via `block_on`; tokens delivered via `mpsc` channel + `spawn_blocking` consumer |
| GIL handling | `#[cfg(Py_GIL_DISABLED)]` for CPython 3.14t (no GIL); `PyEval_SaveThread/RestoreThread` for classic GIL Python |
| `CompletionOutcome` | `.content`, `.rounds`, `.usage` |
| `StreamOutcome` | `.content`, `.finish_reason`, `.usage` |

## High-level diagram

```mermaid
flowchart LR
  subgraph py [Python_3.14t]
    App[user_code]
    Client[superglue_py.Client]
    ToolFn[def_tool_fn]
  end
  subgraph core [Tokio_core]
    HTTP[HttpClient_SSE]
    Loop[complete_with_tools]
    Stream[stream_complete]
    Registry[ToolRegistry]
    Proto[protobuf_types]
  end
  subgraph provider [LLM_Provider]
    API[OpenAI_compatible_API]
  end
  App --> Client
  Client -->|block_on| Loop
  Client -->|block_on| Stream
  Loop --> HTTP
  Stream --> HTTP
  HTTP --> API
  Loop -->|spawn_blocking| ToolFn
  Registry --> Loop
  Proto --> Loop
  Proto --> Stream
```

## Interaction surfaces

| Mode | Status | Notes |
|------|--------|-------|
| **In-process FFI (PyO3)** | ✅ shipped | `superglue-py`; targets CPython 3.14t free-threaded |
| **gRPC / tonic** | 🔲 planned | Would enable polyglot bindings without FFI per language |
| **napi-rs (Node.js)** | 🔲 planned | Same Rust core, different binding layer |

## Token delivery in streaming

Token delivery from the Rust async loop to a Python callback requires care because `Python::attach` must not be called from Tokio's async polling context (segfault on free-threaded CPython). The implemented solution:

1. `stream_complete` calls `on_delta: FnMut(String)` synchronously for each token.
2. In the Python binding, `on_delta` sends to a `tokio::sync::mpsc::unbounded_channel`.
3. A `spawn_blocking` consumer reads from the channel and calls `Python::attach` safely on a dedicated OS thread.
4. `block_on` waits for both the stream and the consumer to finish before returning to Python.
