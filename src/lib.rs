//! Superglue — polyglot LLM orchestration core (WIP).
//!
//! Phase 1: HTTP client with retries, rate limiting, and SSE framing. See [`http`].
//! Phase 2: JSON [`tools`] registry, async [`Tool`] trait, and harness for scripted plans.
//! Phase 2b: OpenAI-shaped [`openai`] types and [`chat`] completions with tool loop.
//! Phase 3: Protobuf canonical schema at [`proto`] — the public API surface for all bindings.
//! Phase 4: Streaming completions via [`chat::stream_complete`] (SSE / `stream: true`).
//! Phase 5: Concurrent [`batch`] completions with configurable error strategies.

pub mod batch;
pub mod chat;
pub mod guardrails;
pub mod hooks;
pub mod http;
pub mod openai;
pub mod proto;
pub mod tools;

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
