//! Cooperative cancellation via [`tokio_util::sync::CancellationToken`].
//!
//! Re-exports the token type so callers don't need to depend on `tokio-util` directly.
//!
//! # Usage
//!
//! ```rust,no_run
//! use superglue::cancel::CancellationToken;
//! use superglue::batch::BatchConfig;
//!
//! let token = CancellationToken::new();
//! let child = token.child_token();
//!
//! // Cancel all in-flight batch requests after 5 seconds.
//! tokio::spawn(async move {
//!     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
//!     token.cancel();
//! });
//!
//! let config = BatchConfig {
//!     cancel: Some(child),
//!     ..BatchConfig::default()
//! };
//! ```

pub use tokio_util::sync::CancellationToken;
