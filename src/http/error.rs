use std::sync::OnceLock;
use std::time::Duration;

use regex::Regex;
use reqwest::StatusCode;
use thiserror::Error;

/// Phase where an LLM stream stopped producing bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamTimeoutPhase {
    /// The response headers or first body chunk did not arrive in time.
    FirstByte,
    /// The established stream produced no bytes within the idle window.
    Idle,
}

impl std::fmt::Display for StreamTimeoutPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstByte => f.write_str("first byte"),
            Self::Idle => f.write_str("idle"),
        }
    }
}

/// Errors from the HTTP layer (transport, status, SSE framing).
#[derive(Debug, Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("HTTP stream timed out during {phase} phase")]
    StreamTimeout { phase: StreamTimeoutPhase },

    #[error("HTTP {status}: body truncated ({len} bytes): {preview}")]
    Unsuccessful {
        status: StatusCode,
        preview: String,
        len: usize,
        /// From the `Retry-After` header when parseable as a delay in seconds (HTTP-date not parsed).
        retry_after: Option<Duration>,
    },

    #[error("retry budget exhausted after {attempts} attempts: {last}")]
    RetriesExhausted { attempts: u32, last: Box<Self> },

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
            Error::StreamTimeout { .. } => true,
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
            Error::StreamTimeout { .. } => true,
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
        let raw = String::from_utf8_lossy(body_sample);
        let preview = redact_sensitive_text(&raw).chars().take(MAX).collect();
        let len = body_sample.len();
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
        match self {
            Self::Unsuccessful { preview, .. } => parse_provider_error_message(preview),
            Self::RetriesExhausted { last, .. } => last.provider_message(),
            _ => None,
        }
    }

    /// Return the HTTP status carried by this error, including its final retry cause.
    #[must_use]
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Reqwest(error) => error.status(),
            Self::Unsuccessful { status, .. } => Some(*status),
            Self::RetriesExhausted { last, .. } => last.status(),
            _ => None,
        }
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
    Some(redact_sensitive_text(&format!("{ty}: {msg}")))
}

fn redact_sensitive_text(text: &str) -> String {
    static REDACT_RE: OnceLock<Regex> = OnceLock::new();
    let pattern = REDACT_RE.get_or_init(|| {
        Regex::new(
            r#"(?i)((?:authorization|api[-_]?key|token|password|secret)\s*[:=]\s*(?:bearer\s+)?|bearer\s+)(["']?)[^"'\s,}]+(["']?)"#,
        )
        .expect("redaction regex must compile")
    });
    pattern
        .replace_all(text, "${1}${2}[REDACTED]${3}")
        .into_owned()
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

    #[test]
    fn retries_exhausted_preserves_provider_error() {
        let last = Error::unsuccessful(
            StatusCode::BAD_GATEWAY,
            br#"{"error":{"type":"upstream_error","message":"model unavailable"}}"#,
            None,
        );
        let error = Error::RetriesExhausted {
            attempts: 4,
            last: Box::new(last),
        };

        assert_eq!(error.status(), Some(StatusCode::BAD_GATEWAY));
        assert_eq!(
            error.provider_message().as_deref(),
            Some("upstream_error: model unavailable")
        );
        assert!(error.to_string().contains("after 4 attempts"));
        assert!(error.to_string().contains("model unavailable"));
    }

    #[test]
    fn error_preview_redacts_credentials() {
        let error = Error::unsuccessful(
            StatusCode::BAD_GATEWAY,
            br#"{"error":{"message":"Authorization: Bearer secret-token api_key=secret-key"}}"#,
            None,
        );
        let display = error.to_string();

        assert!(!display.contains("secret-token"));
        assert!(!display.contains("secret-key"));
        assert!(display.contains("[REDACTED]"));
    }
}
