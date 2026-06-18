//! Superglue — polyglot LLM orchestration core.
//!
//! Phase 1: HTTP client with retries, rate limiting, and SSE framing. See [`http`].
//! Phase 2: JSON [`tools`] registry, async [`Tool`] trait, and harness for scripted plans.
//! Phase 2b: OpenAI-shaped [`openai`] types and [`chat`] completions with tool loop.
//! Phase 3: Protobuf canonical schema at [`proto`] — the public API surface for all bindings.
//! Phase 4: Streaming completions via [`chat::stream_complete`] (SSE / `stream: true`).
//! Phase 5: Concurrent [`batch`] completions with configurable error strategies.
//! Phase 6: [`agents`] with persona/goals/constraints; [`hooks`] and [`guardrails`] pipeline.
//! Phase 7: [`telemetry`] (subscriber wiring, OTLP, metrics, scrubbing), [`cancel`],
//!          [`http::download`] throttling, per-tool policies, [`audit`] run recording,
//!          and optional [`grpc`] server.

pub mod agents;
pub mod audit;
pub mod batch;
pub mod cancel;
pub mod client;
pub mod chat;
pub mod costing;
pub mod events;
pub mod fallback;
pub mod guardrails;
pub mod hooks;
pub mod http;
pub mod openai;
pub mod proto;
pub mod responses;
pub mod telemetry;
pub mod tools;

#[cfg(feature = "mcp")]
pub mod mcp;

pub use client::{CallOptions, Client, ClientBuilder, ClientBuildError, ClientConversation};

#[cfg(feature = "grpc")]
pub mod grpc;

/// Returns a short version string for the crate.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Core greeting used by the CLI `hello` command.
#[must_use]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}!")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn greet_includes_name() {
        assert_eq!(greet("world"), "Hello, world!");
    }
}
