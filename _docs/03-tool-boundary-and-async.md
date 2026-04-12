# Tool boundary and async

## Contract

**Decided direction**: the Rust core treats tools as abstract callbacks keyed by name with **JSON arguments** and **JSON results**. The core does not parse Python type objects or TypeScript interfaces. Each binding is responsible for:

1. Building a **machine-readable schema** (typically JSON Schema) that describes parameters and documentation strings shown to the model.
2. Dispatching execution when the core requests a tool call, including deserialization of arguments and serialization of results to JSON.
3. Reporting failures in a structured way so the core can apply a single policy (retry tool, abort run, surface error to user).

This keeps the orchestration loop **easy to test** in Rust with snapshot-style JSON fixtures.

## Async model

**Decided direction**: orchestration is **async everywhere** in the core (Tokio). Host languages expose async tools; the binding bridges the host’s async runtime to Tokio.

**Decided direction**: **Tokio owns the primary runtime** for network and orchestration work. Python’s asyncio or Node’s libuv are not the source of truth for scheduling LLM calls or parallel tool execution; they integrate via bridges.

### Python (illustrative, not an API spec)

The expected pattern is that a Python async function is registered, and when the core invokes the tool, the PyO3 layer schedules the coroutine and returns a future compatible with Tokio polling (crates such as `pyo3-async-runtimes` exist for this class of problem). Exact crate choice is an **implementation decision** when code exists.

### Errors and cancellation

**Decided direction**: error semantics and cancellation behavior are **defined in the core** so every binding exposes the same outcomes. Bindings translate error types to native exceptions or result types but do not reinterpret policy (for example whether a failed tool aborts the whole run).

**Open decision**: fine-grained per-tool error policies (retry versus fail-fast) may need configuration; capture in the core configuration model when implemented.

## Testing strategy

Unit tests in the core should drive the tool loop with **in-memory JSON** callbacks. Integration tests can use fake LLM responses that emit tool calls, avoiding live network in CI.

## See also

- [Architecture](02-architecture.md) for the split between core and bindings.
- [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md) for how tool events may appear in internal event streams.
