# Vision and principles

## Context

GlueLLM today is a high-level Python SDK (HTTP to model providers, streaming, tools, structured output, batching, retries, hooks, and workflows). The Python implementation in [`../../gluellm/`](../../gluellm/) remains the behavioral reference for "what users expect."

`superglue` is a **Rust reimplementation of that core** — compiled, polyglot, and GIL-free — exposed to Python today via PyO3 and planned for other languages via gRPC or additional FFI bindings.

## Primary goal

Provide a **single, high-quality implementation** of orchestration and provider I/O that can be reused across languages. Fixes to retry logic, tool loops, streaming parsers, and workflow semantics should land once in Rust and propagate everywhere, instead of being reimplemented per SDK.

## Why Rust for the core

- **Performance and concurrency**: network-bound work, concurrent tool execution, and batch scheduling benefit from a Tokio-first design without a Python GIL in the hot path.
- **Deployment flexibility**: a compiled core can ship as a library (PyO3), a sidecar (gRPC), or a small daemon, depending on packaging needs.
- **Ecosystem fit**: mature crates for HTTP (`reqwest`), serialization (`serde`/`prost`), and observability (`tracing`) align with this problem domain.

## Design principles (decided direction)

1. **Orchestration lives in Rust**: the loop that calls LLM APIs, parses SSE streams, decides retries, dispatches tools, and applies workflow rules runs in the core.
2. **Bindings are thin**: host languages supply native ergonomics (registering callables, mapping types, GIL management) but do not reimplement orchestration.
3. **Clear boundaries**: tool arguments and results cross the binding boundary as **JSON**. Internal workflow and hook events use **protobuf** (see [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md)).
4. **Async throughout**: the core is async end-to-end (Tokio). Python bindings block the calling thread with `block_on`; tool callbacks and stream tokens are delivered via `spawn_blocking` so Python's thread state is managed correctly.
5. **Observability is core-owned**: structured traces and metrics originate in Rust via `tracing`; bindings adapt export to host conventions.
6. **Free-threaded Python first**: the Python binding targets CPython 3.14t (`Py_GIL_DISABLED`) to allow true concurrent completions without GIL contention.

## Non-goals

- **Replicating Python-only ergonomics inside Rust**: Rust will not introspect arbitrary Python functions or Pydantic models natively. The Python layer supplies JSON Schema; the core consumes JSON.
- **A single document that specifies every future API**: these notes frame architecture; concrete APIs belong in code and separate API references.

## Open decisions

- Exact **deployment topology** for each language (in-process FFI versus local gRPC sidecar) may vary by customer and platform; see [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md).
- **Licensing and distribution** (open core versus proprietary daemon) are business choices not fixed here.
