use std::time::Duration;

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
        /// From the `Retry-After` header when parseable as a delay in seconds (HTTP-date not parsed).
        retry_after: Option<Duration>,
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

    pub(crate) fn unsuccessful(
        status: StatusCode,
        body_sample: &[u8],
        retry_after: Option<Duration>,
    ) -> Self {
        const MAX: usize = 512;
        let len = body_sample.len();
        let preview = String::from_utf8_lossy(&body_sample[..len.min(MAX)]).into_owned();
        Self::Unsuccessful {
            status,
            preview,
            len,
            retry_after,
        }
    }

    /// Parse a provider JSON error body when present.
    #[must_use]
    pub fn provider_message(&self) -> Option<String> {
        let Self::Unsuccessful { preview, .. } = self else {
            return None;
        };
        parse_provider_error_message(preview)
    }

    /// When the server returned `429` with `Retry-After: N` (seconds), prefer this wait duration.
    #[must_use]
    pub fn retry_after_hint(&self) -> Option<Duration> {
        match self {
            Self::Unsuccessful {
                status,
                retry_after,
                ..
            } if *status == StatusCode::TOO_MANY_REQUESTS => *retry_after,
            _ => None,
        }
    }
}

fn parse_provider_error_message(body: &str) -> Option<String> {
    let start = body.find('{')?;
    let value: serde_json::Value = serde_json::from_str(&body[start..]).ok()?;
    let err = value.get("error")?;
    let ty = err.get("type").and_then(|v| v.as_str())?;
    let msg = err
        .get("message")
        .and_then(|v| v.as_str())
        .filter(|m| !m.is_empty())
        .unwrap_or("request failed");
    Some(format!("{ty}: {msg}"))
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn parses_anthropic_error_json() {
        let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limit reached"}}"#;
        assert_eq!(
            parse_provider_error_message(body),
            Some("rate_limit_error: Rate limit reached".to_string())
        );
    }
}
