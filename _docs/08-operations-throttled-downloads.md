# Operations: throttled downloads

## Use cases

Large artifacts (for example **local model files**) may need to be downloaded without saturating shared networks or triggering provider rate limits on parallel hosts. The conversation called out **fixed effective download speed** as a control knob.

## Approaches

### Chunk pacing

Read the HTTP response stream in chunks, write each chunk, then **sleep** to maintain an average bytes-per-second target. Simple implementations drift over long transfers because scheduling and OS buffering are imperfect.

### Token bucket

A **token bucket** rate limiter enforces smoother throughput and composes with other limiters. In Rust, **`governor`** is a common choice for production-grade token buckets and aligns conceptually with **API rate limiting** for LLM calls elsewhere in the stack.

Using the **same abstraction family** for “download bandwidth” and “provider QPS” reduces cognitive load for operators, even if the parameters differ.

## Implementation notes (non-binding)

- Prefer **async** reads and writes (`tokio` I/O) so throttling does not block unrelated tasks on the runtime.
- Surface **progress** and **effective throughput** in logs or UI where product requirements call for it (for example `indicatif` in CLIs).

## Relation to the core roadmap

Bandwidth throttling is **orthogonal** to the LLM orchestration MVP. Schedule it with operational features (CLI, daemon updates) rather than blocking the first streaming HTTP client milestone. See [Roadmap and phasing](09-roadmap-and-phasing.md).

## See also

- [Ecosystem reference](10-ecosystem-reference.md) for crates commonly used alongside Tokio HTTP stacks.
