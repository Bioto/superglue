# Observability

## Goals

Provide **one implementation** of instrumentation inside the Rust core so that latency, token usage, retries, tool invocations, and workflow steps are measured consistently. Language bindings should **adapt** signals to host-native logging and tracing systems, not reimplement metrics.

## Decided direction

- **Tracing**: use the `tracing` ecosystem with async-aware spans around LLM requests, stream chunks, tool calls, retries, and workflow transitions.
- **Export**: support OpenTelemetry via `tracing-opentelemetry` (or equivalent) so spans can reach OTLP collectors. Exact feature flags and configuration are implementation details.
- **Metrics**: use a metrics facade (for example the `metrics` crate family) for counters and histograms such as request duration, queue depth, and retry counts.

## Redaction and scrubbing

**Design constraint (not optional at ship time)**: raw prompts, tool arguments, tool outputs, and provider responses may contain secrets or PII. Before any log line, trace attribute, or metric label is exported, data must pass through a **scrubbing or redaction layer**.

Policies should include:

- Default **deny** for full payloads in spans and logs.
- Opt-in **sampled** or **hashed** representations for debugging sessions.
- Explicit lists of **safe** attributes (for example provider name, model id, HTTP status class) versus **unsafe** fields that must never attach to OTLP without transformation.

Bindings that add host-side logging must follow the same rules so that Python or Node cannot accidentally bypass core policy.

## Errors

Structured error types should preserve **context chains** (provider error, parse error, tool error) in Rust. Bindings map these to native error types while preserving stable **error codes** where feasible for programmatic handling.

## Relation to security posture

Telemetry is a **major leak vector** for sensitive data. See [Security and threat model](06-security-and-threat-model.md) for alignment with “leave no trace” requirements when that tier applies.

## See also

- [Architecture](02-architecture.md) for what runs inside the core versus bindings.
- [Network privacy and telemetry shaping](07-network-privacy-and-telemetry-shaping.md) for optional traffic-level considerations separate from application-level redaction.
