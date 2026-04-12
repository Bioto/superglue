# Superglue design documentation

This directory holds **maintainer-facing design notes** for a future polyglot orchestration layer related to the ideas behind [GlueLLM](https://gluellm.dev). The experimental Rust crate in this repository is [`superglue`](../); the current Python reference implementation lives in [`../../gluellm/`](../../gluellm/).

These documents describe **intent and tradeoffs**, not a shipped API. They are versioned with the repo and should be updated as decisions change.

## Reading order

1. [Vision and principles](01-vision-and-principles.md) — goals, constraints, non-goals.
2. [Architecture](02-architecture.md) — what runs in the Rust core versus language bindings.
3. [Tool boundary and async](03-tool-boundary-and-async.md) — JSON tool payloads, Tokio ownership, bridging async hosts.
4. [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md) — internal schema, RPC surface, deployment modes.
5. [Observability](05-observability.md) — tracing, metrics, and safe export of telemetry.
6. [Security and threat model](06-security-and-threat-model.md) — tiered posture from commercial packaging to high-assurance patterns (with honest limits).
7. [Network privacy and telemetry shaping](07-network-privacy-and-telemetry-shaping.md) — optional traffic-shaping ideas; legal and ethical caveats.
8. [Operations: throttled downloads](08-operations-throttled-downloads.md) — rate-limited artifact fetch (for example model files).
9. [Roadmap and phasing](09-roadmap-and-phasing.md) — incremental delivery plan.
10. [Ecosystem reference](10-ecosystem-reference.md) — Rust crates and prior art to study.

## How to use this corpus

- **Decided direction** is stated explicitly where the group has converged (for example: orchestration and I/O in Rust, JSON across the tool boundary).
- **Open decisions** are called out as questions or options rather than pretending they are settled.
- **Exploratory or adversarial** topics (for example obfuscation or self-destruct flows) are isolated in dedicated sections so they do not read as default product requirements.

## Related code in this repository

| Area | Location |
|------|----------|
| Python GlueLLM (behavioral reference) | [`../../gluellm/`](../../gluellm/) |
| Rust crate scaffold | [`../`](../) |
