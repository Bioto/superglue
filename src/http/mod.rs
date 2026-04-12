//! HTTP transport: timeouts, retries, optional QPS limiting, SSE helpers.
//!
//! Phase 1 building block for provider calls and streaming responses.

mod client;
mod error;
mod rate_limit;
mod retry;
pub mod sse;
pub mod url;

pub use client::{ClientConfig, HttpClient};
pub use error::Error;
pub use retry::RetryPolicy;
pub use sse::{SseEvent, SseParser};
pub use url::join_base_url;
