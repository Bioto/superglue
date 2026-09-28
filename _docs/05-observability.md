# Observability

## Goals

Provide **one implementation** of instrumentation inside the Rust core so that latency, token usage, retries, tool invocations, and workflow steps are measured consistently. Language bindings should **adapt** signals to host-native logging and tracing systems, not reimplement metrics.

## Decided direction

- **Tracing**: use the `tracing` ecosystem with async-aware spans around LLM requests, stream chunks, tool calls, retries, and workflow transitions.
- **Process events**: optional [`StatusEmitter`](../src/events/mod.rs) fan-out emits typed `ProcessEvent` records (`llm_call_start`, `llm_call_end`, `llm_call_error`, `tool_call_*`) with token usage and estimated USD cost. Opt-in via `ChatOptions.status_emitter` or Python `Client(status_emitter=...)`.
- **Hooks vs events**: hooks may mutate pipeline content; status events are observation-only.
- **Export**: support OpenTelemetry via `tracing-opentelemetry` so spans can reach OTLP collectors. Enable the `otlp` feature and set `TelemetryConfig::otlp_endpoint`.
- **Span meaning**: exported spans use [OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md) names.
  - `openinference.span.kind` is `AGENT`, `CHAIN`, `LLM`, `TOOL`, or `GUARDRAIL`.
  - An LLM span carries `llm.model_name`, `llm.provider`, `llm.system`, `llm.invocation_parameters`, flattened `llm.input_messages` / `llm.output_messages`, and `llm.token_count.*`.
  - A tool span carries `tool.name`, `tool.id`, `input.value`, and `output.value`.
  - An agent span carries stable `agent.id` and `agent.name` from `AgentSpec::name`, plus `session.id`.
  - `session.id` is `ChatOptions.request_id`. Set that id on the caller when one conversation should stay together.
- **Metrics**: use a metrics facade (for example the `metrics` crate family) for counters and histograms such as request duration, queue depth, and retry counts.

## Redaction and scrubbing

**Design constraint (not optional at ship time)**: raw prompts, tool arguments, tool outputs, and provider responses may contain secrets or PII. Before any log line, trace attribute, or metric label is exported, data must pass through a **scrubbing or redaction layer**.

Policies should include:

- Default **deny** for full payloads in spans and logs. `ScrubMode::Redact` writes `[REDACTED]` into OpenInference message, tool-argument, and tool-result attributes. `ScrubMode::Allow` writes the text. `data:` URLs stay `[data-uri]` in every mode.
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
