# Roadmap and phasing

## Principles

- **Avoid a top-down rewrite**: ship narrow vertical slices that are testable and usable, then expand.
- **Behavioral parity** with [`../../gluellm/`](../../gluellm/) as a guiding star; divergence is intentional and documented.
- **Real usage drives priorities**: the Python binding is the first external-facing surface.

---

## Phase 1: HTTP, retries, streaming ✅ complete

- ✅ `HttpClient` — authenticated HTTP requests (`reqwest`, `rustls-tls`).
- ✅ Streaming responses via SSE framing (`SseParser`).
- ✅ Timeouts, retries (GET and POST with distinct policies), and rate limits (`governor`).
- ✅ Deterministic wiremock integration tests.

## Phase 2: Tool boundary and Python binding ✅ complete

- ✅ `Tool` trait + `ToolRegistry` — async dispatch by name with JSON arguments/results.
- ✅ `chat::complete_with_tools` — multi-turn tool loop against any OpenAI-compatible provider.
- ✅ `tools::harness` — offline scripted execution plans for unit testing tool flows.
- ✅ `superglue-py` Python binding (PyO3, maturin, CPython 3.14t free-threaded).
- ✅ `Client.complete()` with Python dict tool callbacks delivered via `spawn_blocking`.
- ✅ `Client.stream()` with per-token callback delivered via `mpsc` channel + `spawn_blocking`.
- ✅ 14 Python example scripts covering: basic completion, system prompts, single/multiple tools, usage tracking, error handling, max rounds, custom base URL, concurrent completions, structured tool data, and streaming variants.

## Phase 3: Full OpenAI spec + protobuf schema ✅ complete

- ✅ Full `ChatCompletionRequest` — all OpenAI parameters (sampling, penalties, stop, response format, tool choice, logprobs, seed, store, service tier, reasoning effort, stream options).
- ✅ Multipart message content (`MessageContent`: text, image_url, input_audio, file).
- ✅ Complete response types — `Usage` with token details, `ChoiceLogprobs`, `Annotation`, `ChatCompletionChunk` for streaming.
- ✅ `ChatOptions` exposes all key parameters; `From<proto::ChatOptions>` bridge.
- ✅ `proto/superglue.proto` schema covering `ChatOptions`, `ChatMessage`, `Usage`, `ToolSpec`, `CompletionOutcome`.
- ✅ `prost` + `protoc-bin-vendored` code generation via `build.rs`.
- ✅ `stream_complete` — SSE streaming with usage in final chunk (`stream_options.include_usage`).

## Phase 4: gRPC and additional bindings ✅ partial

- ✅ `tonic` gRPC server exposing `complete` (unary) and `stream` (server-streaming) as RPCs.
- ✅ Audit trail — `HookEvent` + `RunRecord` proto types; `RunStore` + `RunRecorder`.
- 🔲 Proto-generated client stubs for Python (replacing PyO3 for the gRPC deployment mode).
- 🔲 Node.js binding via `napi-rs`.
- 🔲 Workflow engine and hook dispatch with protobuf event streams.
- 🔲 Replay / run-resume from recorded protobuf audit streams.

## Phase 5: Hardening and ecosystem ✅ partial

- ✅ Structured redaction and observability tiers — `ScrubMode` (Redact / Hash / Allow), OTLP export (`otlp` feature) (see [Observability](05-observability.md)).
- ✅ Security hardening (Tier B) — `secrecy::Secret<String>` for API keys, `zeroize` on drop (see [Security and threat model](06-security-and-threat-model.md)).
- ✅ Batch scheduling and parallel tool execution within a single turn (`join_all`).
- ✅ Per-tool error policies (FailFast / Skip / Retry with exponential back-off).
- ✅ Cooperative cancellation (`CancellationToken`) in batch and chat loops.
- ✅ Throttled downloads (`governor` bandwidth limiter with progress callback).
- ✅ Context optimization (GlueLLM parity): dynamic tool routing, tool-round condensing, AAAK encoding, history compression — see [Context optimization](11-context-optimization.md).
- 🔲 Migration guide from Python GlueLLM (partial: context optimization flags documented in [11-context-optimization.md](11-context-optimization.md)).

## Continuous cross-cutting work

- **Observability**: `tracing` spans are present throughout; exporter wiring (`tracing-subscriber`, OTLP) can land any time.
- **Testing**: wiremock integration tests gate streaming, tool loops, system prompts, and multipart content.
- **Docs**: this directory tracks implementation status and is updated with each phase.
