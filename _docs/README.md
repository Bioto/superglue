# Superglue design documentation

This directory holds **maintainer-facing design notes** for the `superglue` polyglot LLM orchestration core. The Rust crate lives at [`../`](../); the Python binding lives at [`../../superglue-py/`](../../superglue-py/).

These documents describe **intent, tradeoffs, and current implementation status**. They are versioned with the repo and updated as decisions change.

## Reading order

1. [Vision and principles](01-vision-and-principles.md) — goals, constraints, non-goals.
2. [Architecture](02-architecture.md) — what runs in the Rust core versus language bindings.
3. [Tool boundary and async](03-tool-boundary-and-async.md) — JSON tool payloads, Tokio ownership, bridging async hosts.
4. [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md) — internal schema, RPC surface, deployment modes.
5. [Observability](05-observability.md) — tracing, metrics, and safe export of telemetry.
6. [Security and threat model](06-security-and-threat-model.md) — tiered posture from commercial packaging to high-assurance patterns.
7. [Network privacy and telemetry shaping](07-network-privacy-and-telemetry-shaping.md) — optional traffic-shaping ideas; legal and ethical caveats.
8. [Operations: throttled downloads](08-operations-throttled-downloads.md) — rate-limited artifact fetch.
9. [Roadmap and phasing](09-roadmap-and-phasing.md) — incremental delivery plan with current status.
10. [Ecosystem reference](10-ecosystem-reference.md) — Rust crates in use and prior art.

## Current implementation status

| Component | Status |
|-----------|--------|
| HTTP client (`reqwest`, rustls-tls) | ✅ shipped |
| Retry policy (GET + POST with distinct rules) | ✅ shipped |
| Rate limiting (`governor` token bucket) | ✅ shipped |
| SSE parser (`SseParser`) | ✅ shipped |
| Tool registry + `Tool` trait | ✅ shipped |
| Tool harness (offline scripted plans) | ✅ shipped |
| OpenAI chat types (full spec) | ✅ shipped |
| Chat completion tool loop (`complete_with_tools`) | ✅ shipped |
| Parallel tool dispatch within a round | ✅ shipped |
| Streaming completions (`stream_complete`, SSE) | ✅ shipped |
| Protobuf schema (`prost`, `proto/superglue.proto`) | ✅ shipped |
| Python binding (`superglue-py`, PyO3, CPython 3.14t) | ✅ shipped |
| Python `Client.complete()` with tool callbacks | ✅ shipped |
| Python `Client.stream()` with token callback | ✅ shipped (GIL-free 3.14t) |
| Per-tool error policies (FailFast / Skip / Retry) | ✅ shipped |
| Cooperative cancellation (`CancellationToken`) | ✅ shipped |
| Throttled downloads (`governor` bandwidth limiter) | ✅ shipped |
| API key zeroization (`secrecy` + `zeroize`) | ✅ shipped |
| Observability — tracing scrub layer + metrics facade | ✅ shipped |
| Observability — OTLP export (`otlp` feature) | ✅ shipped |
| Observability — configurable scrub modes (Redact/Hash/Allow) | ✅ shipped |
| Audit trail (`RunStore` + `RunRecorder`) | ✅ shipped |
| gRPC server (`tonic`, `grpc` feature) | ✅ shipped |
| Node.js binding (`napi-rs`) | 🔲 planned |
| Proto-generated language client stubs | 🔲 planned |
| Replay / run-resume from audit trail | 🔲 planned |
| Workflow engine | 🔲 planned |

## Related code

| Area | Location |
|------|----------|
| Rust core | [`../`](../) |
| Python binding | [`../../superglue-py/`](../../superglue-py/) |
| Python examples | [`../../superglue-py/examples/`](../../superglue-py/examples/) |
| Python GlueLLM (behavioral reference) | [`../../gluellm/`](../../gluellm/) |
