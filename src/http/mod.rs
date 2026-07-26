//! HTTP transport: timeouts, retries, optional QPS limiting, SSE helpers, and throttled downloads.
//!
//! Phase 1 building block for provider calls and streaming responses.
//! Phase N: [`download`] module adds token-bucket-throttled artifact fetching.

mod client;
pub mod download;
mod error;
mod rate_limit;
mod retry;
pub mod sse;
pub mod url;

pub use client::{ClientConfig, HttpClient};
pub use download::{DownloadConfig, download_throttled};
pub use error::Error;
pub use rate_limit::{DirectRateLimiter, direct_per_second};
pub use retry::RetryPolicy;
pub use sse::{SseEvent, SseParser};
pub use url::join_base_url;
