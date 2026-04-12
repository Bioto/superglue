# Ecosystem reference

This is a **reading list and crate map**, not a dependency manifest. Versions belong in `Cargo.toml` when implementation starts.

## HTTP and streaming

- **`reqwest`**: async HTTP client on Tokio; common baseline for provider APIs.
- **`tower`**: middleware abstraction; composes retries, timeouts, and rate limits; aligns with `tonic` stacks.
- **`eventsource` / stream utilities**: ecosystem crates for SSE vary; study **`async-openai`** (read the source) for streaming patterns even if you do not depend on it directly.

## Serialization

- **`serde`**, **`serde_json`**: JSON tool payloads and provider JSON.
- **`prost`**, **`prost-build`**: protobuf code generation in Rust.
- **`tonic`**: gRPC on Tokio with `prost` messages.

## Async and control flow

- **`tokio`**: runtime, timers, async I/O.
- **`tokio-util`**: helpers such as cancellation tokens (`CancellationToken`) for cooperative shutdown.
- **`backoff`** or **`tower::retry`**: policy for retries depending on stack choice.

## Rate limiting

- **`governor`**: token bucket rate limiting; conceptually reusable for downloads (see [Operations: throttled downloads](08-operations-throttled-downloads.md)) and API quotas.

## Observability

- **`tracing`**, **`tracing-subscriber`**: structured logs and spans.
- **`tracing-opentelemetry`**: bridge spans to OTLP and other exporters.
- **`metrics`**: metrics facade; pick an exporter that matches deployment.

## Security utilities (when needed)

- **`rustls`**: TLS without linking OpenSSL; pairs with `tonic` transports that support it.
- **`zeroize`**, **`secrecy`**: sensitive memory handling (see [Security and threat model](06-security-and-threat-model.md)).
- **`ring`**: cryptographic primitives when implementing integrity checks or custom crypto (review carefully; prefer high-level patterns where possible).

## Language bindings (study, not immediate deps of `superglue` crate)

- **`pyo3`**, **`maturin`**: Python extension modules.
- **`pyo3-async-runtimes`** (or successors): bridge asyncio and Tokio.
- **`napi-rs`**: Node.js native addons if FFI to Node is required.

Prior art for **how** to structure native cores with Python wrappers: inspect **`ruff`** and **`uv`** repositories for packaging and PyO3 layout lessons.

## Testing CLI and processes

- **`assert_cmd`**, **`predicates`**: integration tests invoking compiled binaries (already used in the `superglue` scaffold tests).

## See also

- [Roadmap and phasing](09-roadmap-and-phasing.md) for when to introduce each layer.
- [Architecture](02-architecture.md) for how these pieces fit together.
