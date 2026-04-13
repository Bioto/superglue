//! Throttled artifact downloads.
//!
//! [`download_throttled`] streams an HTTP response to a file at a configurable
//! maximum byte rate, using the same `governor` token-bucket abstraction as
//! the API rate limiter so operators only need to reason about one abstraction.
//!
//! # Example
//!
//! ```rust,no_run
//! use superglue::http::{ClientConfig, HttpClient, download::{DownloadConfig, download_throttled}};
//! use std::path::Path;
//!
//! async fn example() {
//!     let http = HttpClient::new(ClientConfig::default()).unwrap();
//!     let bytes = download_throttled(
//!         &http,
//!         "https://example.com/model.bin",
//!         Path::new("/tmp/model.bin"),
//!         DownloadConfig {
//!             max_bytes_per_sec: 1024 * 1024, // 1 MiB/s
//!             on_progress: Some(Box::new(|downloaded, total| {
//!                 if let Some(t) = total {
//!                     println!("{downloaded}/{t} bytes");
//!                 }
//!             })),
//!         },
//!     )
//!     .await
//!     .unwrap();
//!     println!("downloaded {bytes} bytes");
//! }
//! ```

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use futures_util::StreamExt;
use governor::{Quota, RateLimiter, clock::DefaultClock, state::InMemoryState, state::NotKeyed, middleware::NoOpMiddleware};
use tokio::io::AsyncWriteExt;
use tracing::instrument;

use super::{Error, HttpClient};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`download_throttled`].
pub struct DownloadConfig {
    /// Maximum byte throughput in bytes per second.
    ///
    /// `0` means unlimited — the download proceeds as fast as the server and
    /// network allow.
    pub max_bytes_per_sec: u64,

    /// Optional progress callback.
    ///
    /// Called after each chunk is written with `(bytes_written_so_far, content_length)`.
    /// `content_length` is `None` when the server does not send `Content-Length`.
    pub on_progress: Option<Box<dyn Fn(u64, Option<u64>) + Send>>,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        DownloadConfig {
            max_bytes_per_sec: 0,
            on_progress: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Type alias for the byte-rate limiter
// ---------------------------------------------------------------------------

type ByteLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock, NoOpMiddleware>;

fn make_byte_limiter(bytes_per_sec: u64) -> Arc<ByteLimiter> {
    // governor works in tokens-per-second; we use 1 token ≈ 1 KiB to keep
    // the quota within u32 limits (max ~4 TiB/s in theory).
    // The quota is rounded up to at least 1.
    let kib_per_sec = ((bytes_per_sec + 1023) / 1024).max(1).min(u32::MAX as u64) as u32;
    let quota = Quota::per_second(NonZeroU32::new(kib_per_sec).expect("kib_per_sec > 0"));
    Arc::new(RateLimiter::direct(quota))
}

// ---------------------------------------------------------------------------
// Core function
// ---------------------------------------------------------------------------

/// Download `url` to `dest`, throttling throughput to at most
/// `config.max_bytes_per_sec` bytes per second.
///
/// Returns the total number of bytes written on success.
///
/// # Errors
///
/// Returns [`Error`] on HTTP or I/O failure. On non-2xx status the download
/// is aborted immediately with [`Error::Unsuccessful`].
#[instrument(skip(http, config), fields(url = url))]
pub async fn download_throttled(
    http: &HttpClient,
    url: &str,
    dest: &Path,
    config: DownloadConfig,
) -> Result<u64, Error> {
    let limiter: Option<Arc<ByteLimiter>> = if config.max_bytes_per_sec > 0 {
        Some(make_byte_limiter(config.max_bytes_per_sec))
    } else {
        None
    };

    let mut stream = http.get_stream(url).await?;

    // Try to get Content-Length from a HEAD request is impractical here; the
    // get_stream() response doesn't expose headers. We leave total as None for now.
    let content_length: Option<u64> = None;

    let file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| Error::InvalidJson(format!("create {}: {e}", dest.display())))?;
    let mut writer = tokio::io::BufWriter::new(file);

    let mut total_written: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        let len = bytes.len() as u64;

        if let Some(lim) = &limiter {
            // Convert chunk bytes to KiB tokens (1 KiB per token, rounded up, min 1).
            let kib_tokens = ((bytes.len() + 1023) / 1024).max(1);
            let mut remaining = kib_tokens;
            while remaining > 0 {
                // Request at most 1 KiB at a time to stay within burst capacity.
                let tokens = NonZeroU32::new(1).unwrap();
                // until_n_ready is async; if the capacity is exceeded it returns
                // an error immediately, so we fall back to single-token waits.
                match lim.until_n_ready(tokens).await {
                    Ok(()) => {}
                    Err(_insufficient) => {
                        // Bucket capacity < requested tokens: just wait a tick.
                        lim.until_ready().await;
                    }
                }
                remaining -= 1;
            }
        }

        writer
            .write_all(&bytes)
            .await
            .map_err(|e| Error::InvalidJson(format!("write: {e}")))?;

        total_written += len;

        if let Some(cb) = &config.on_progress {
            cb(total_written, content_length);
        }
    }

    writer
        .flush()
        .await
        .map_err(|e| Error::InvalidJson(format!("flush: {e}")))?;

    tracing::info!(url, bytes = total_written, "download_throttled complete");
    Ok(total_written)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_builds_without_panic() {
        let _ = make_byte_limiter(1024 * 1024);
        let _ = make_byte_limiter(1);
    }

    #[test]
    fn download_config_default_is_unlimited() {
        let c = DownloadConfig::default();
        assert_eq!(c.max_bytes_per_sec, 0);
        assert!(c.on_progress.is_none());
    }
}
