# Vision and principles

## Context

GlueLLM today is a high-level Python SDK (HTTP to model providers, streaming, tools, structured output, batching, retries, hooks, and workflows). The Python implementation in [`../../gluellm/`](../../gluellm/) remains the behavioral reference for “what users expect,” even as a Rust-centered design evolves under `superglue`.

## Primary goal

Provide a **single, high-quality implementation** of orchestration and provider I/O that can be reused across languages. Fixes to retry logic, tool loops, streaming parsers, and workflow semantics should land once and propagate everywhere, instead of being reimplemented per SDK.

## Why Rust for the core

- **Performance and concurrency**: network-bound work, concurrent tool execution, and batch scheduling benefit from a Tokio-first design without a Python GIL in the hot path.
- **Deployment**: a compiled core can ship as a library, a sidecar, or a small daemon, depending on packaging needs.
- **Ecosystem fit**: mature crates for HTTP (`reqwest`), middleware (`tower`), serialization (`serde`), and observability (`tracing`) align with this problem domain.

## Design principles (decided direction)

1. **Orchestration lives in Rust**: the loop that calls LLM APIs, parses streams, decides retries, dispatches tools, and applies workflow rules runs in the core.
2. **Bindings are thin**: host languages supply native ergonomics (for example registering callables and mapping types) but do not reimplement orchestration.
3. **Clear boundaries**: tool arguments and results cross the boundary as **JSON** (see [Tool boundary and async](03-tool-boundary-and-async.md)). Internal workflow and hook events may use **protobuf** and **gRPC** (see [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md)).
4. **Async throughout**: the core is async end-to-end; hosts bridge their runtimes to Tokio rather than forcing the core to mirror each host’s event loop.
5. **Observability is core-owned**: structured traces and metrics originate in Rust; bindings only adapt export to host conventions (see [Observability](05-observability.md)).

## Non-goals

- **Replicating Python-only ergonomics inside Rust**: Rust will not introspect arbitrary Python functions or Pydantic models natively. The Python layer may generate JSON Schema and adapters; the core consumes JSON.
- **A single document that specifies every future API**: these notes frame architecture; concrete APIs belong in code and separate API references when they exist.

## Open decisions

- Exact **deployment topology** for each language (in-process FFI versus local gRPC sidecar) may vary by customer and platform; see [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md).
- **Licensing and distribution** (open core versus proprietary daemon) are business choices not fixed here.
