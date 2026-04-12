use reqwest::StatusCode;
use thiserror::Error;

/// Errors from the HTTP layer (transport, status, SSE framing).
#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("HTTP {status}: body truncated ({len} bytes): {preview}")]
    Unsuccessful {
        status: StatusCode,
        preview: String,
        len: usize,
    },

    #[error("retry budget exhausted after {attempts} attempts (last status: {last:?})")]
    RetriesExhausted {
        attempts: u32,
        last: Option<StatusCode>,
    },

    #[error("invalid SSE line: {0}")]
    SseParse(String),

    #[error("rate limiter unavailable: {0}")]
    RateLimit(String),

    #[error("invalid JSON in response body: {0}")]
    InvalidJson(String),
}

impl Error {
    /// Whether this error is eligible for a fresh retry (transport or retryable HTTP status).
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Reqwest(e) => super::retry::RetryPolicy::is_retryable_reqwest_error(e),
            Error::Unsuccessful { status, .. } => {
                super::retry::RetryPolicy::is_retryable_status(*status)
            }
            _ => false,
        }
    }

    /// Retry policy for **POST** JSON: transport errors and overload-style statuses only (avoids
    /// blindly retrying 500s that may have partially processed the request).
    #[must_use]
    pub fn is_retryable_post(&self) -> bool {
        match self {
            Error::Reqwest(e) => super::retry::RetryPolicy::is_retryable_post_reqwest_error(e),
            Error::Unsuccessful { status, .. } => {
                super::retry::RetryPolicy::is_retryable_post_status(*status)
            }
            _ => false,
        }
    }

    pub(crate) fn unsuccessful(status: StatusCode, body_sample: &[u8]) -> Self {
        const MAX: usize = 512;
        let len = body_sample.len();
        let preview = String::from_utf8_lossy(&body_sample[..len.min(MAX)]).into_owned();
        Self::Unsuccessful {
            status,
            preview,
            len,
        }
    }
}
