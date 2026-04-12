# Ecosystem reference

Crates that are **in use** are marked ✅. Planned or studied crates are marked 🔲.

## HTTP and streaming

| Crate | Status | Role |
|-------|--------|------|
| `reqwest` (rustls-tls, stream, json) | ✅ | Async HTTP client; GET + POST + streaming POST |
| `bytes` | ✅ | Byte buffer type for streaming chunks |
| `futures-util` | ✅ | `StreamExt` for consuming byte streams |
| `tower` | 🔲 | Middleware abstraction; aligns with `tonic` stacks |

SSE parsing is implemented in-house (`src/http/sse.rs`) rather than a crate dependency, to keep the dependency footprint small and the parser fully tested.

## Serialization

| Crate | Status | Role |
|-------|--------|------|
| `serde` + `serde_json` | ✅ | JSON tool payloads and provider JSON |
| `prost` | ✅ | Protobuf runtime for generated types |
| `prost-build` | ✅ | Build-time code generation from `.proto` files |
| `protoc-bin-vendored` | ✅ | Bundles `protoc` so no system install is required |
| `tonic` | 🔲 | gRPC on Tokio; pairs with `prost` messages |

## Async and control flow

| Crate | Status | Role |
|-------|--------|------|
| `tokio` (rt-multi-thread, macros, sync, time, io-util) | ✅ | Runtime, timers, `mpsc` channels, `spawn_blocking` |
| `async-trait` | ✅ | `async fn` in the `Tool` trait object |
| `tokio-util` | 🔲 | `CancellationToken` for cooperative shutdown |

## Rate limiting

| Crate | Status | Role |
|-------|--------|------|
| `governor` | ✅ | Token-bucket rate limiter; optional per-client QPS cap |

## Observability

| Crate | Status | Role |
|-------|--------|------|
| `tracing` | ✅ | Structured spans and events throughout the core |
| `tracing-subscriber` | 🔲 | Subscriber wiring for production deployments |
| `tracing-opentelemetry` | 🔲 | Bridge spans to OTLP exporters |
| `metrics` | 🔲 | Metrics facade |

## Error handling

| Crate | Status | Role |
|-------|--------|------|
| `thiserror` | ✅ | Derives `Error` for `http::Error`, `ChatError`, `ToolInvokeError` |

## Security utilities

| Crate | Status | Role |
|-------|--------|------|
| `rustls` | ✅ | TLS (via `reqwest`'s `rustls-tls` feature; no OpenSSL) |
| `zeroize` / `secrecy` | 🔲 | Sensitive memory handling for API keys |
| `ring` | 🔲 | Cryptographic primitives for integrity checks |

## Language bindings

| Crate | Status | Role |
|-------|--------|------|
| `pyo3` 0.28 | ✅ | Python extension module; `#[pyclass]`, `#[pymethods]` |
| `maturin` | ✅ | Build and packaging tool for the Python wheel |
| `pyo3-async-runtimes` | 🔲 | asyncio ↔ Tokio bridge (not needed for sync callback pattern) |
| `napi-rs` | 🔲 | Node.js native addons |

**Packaging note**: builds target CPython 3.14t (free-threaded, `Py_GIL_DISABLED`). The `dev.sh` script always points maturin at the `.venv/bin/python` interpreter to ensure the correct ABI tag (`cp314t`) is used, then installs the resulting wheel via `uv pip install` so that subsequent `uv run` invocations do not reinstall the package.

## Testing

| Crate | Status | Role |
|-------|--------|------|
| `wiremock` | ✅ | Mock HTTP server for integration tests |
| `assert_cmd` | ✅ | Integration tests invoking compiled binaries (CLI) |
| `predicates` | ✅ | Assertion helpers for `assert_cmd` |

## Prior art studied

- **`async-openai`**: reference for streaming patterns and OpenAI type structures.
- **`ruff`** and **`uv`**: packaging and PyO3 layout lessons for Rust-based Python tooling.

## See also

- [Roadmap and phasing](09-roadmap-and-phasing.md) for when to introduce remaining layers.
- [Architecture](02-architecture.md) for how these pieces fit together.
