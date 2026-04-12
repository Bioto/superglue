# Roadmap and phasing

## Principles

- **Avoid a top-down rewrite** of Python GlueLLM before the Rust core proves value.
- Ship **narrow vertical slices** that are testable and usable, then expand.
- Keep **behavioral parity** with [`../../gluellm/`](../../gluellm/) as a guiding star, documented where divergence is intentional.

## Phase 1: HTTP, retries, streaming

Deliver a Rust module that can:

- Perform authenticated HTTP requests to at least one provider style used today.
- Handle **streaming** responses (including SSE-style framing where applicable).
- Apply **timeouts**, **retries**, and **rate limits** with deterministic tests.

Outcome: foundation for all higher-level features; no requirement for tools or workflows yet.

## Phase 2: Tool callback interface and one binding

Introduce the **JSON tool boundary** (see [Tool boundary and async](03-tool-boundary-and-async.md)) with in-memory tests in Rust.

Add **one** host binding (likely Python via PyO3 **or** a thin gRPC client, depending on packaging choices) that proves:

- Tool registration from the host.
- Async tool execution bridged to Tokio.
- End-to-end completion with at least one tool round-trip.

Outcome: validates FFI or RPC ergonomics before multiplying languages.

## Phase 3: Protobuf schemas and gRPC services

Define `.proto` files for workflows, hooks, and internal events; generate Rust with `prost` and expose servers or bidi streams with `tonic` as needed (see [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md)).

Outcome: stable internal contracts and optional **sidecar** deployments.

## Phase 4: Additional language clients and polish

Generate or hand-write clients for more languages, improve packaging, and harden observability and security tiers per product requirements.

## Continuous cross-cutting work

- **Observability** and **redaction** should land early enough that early adopters never rely on unsafe logging defaults (see [Observability](05-observability.md)).
- **Documentation** and migration guides from Python GlueLLM should track Phase 2 and beyond.

## Open sequencing

Exact ordering between **Phase 2** and **Phase 3** may shift if gRPC-first deployment wins for a major customer; the phases are **logical** rather than calendar promises.
