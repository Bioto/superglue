use reqwest::StatusCode;

/// Retry policy for idempotent requests (GET) and retryable failures.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_interval_ms: u64,
    pub max_interval_ms: u64,
    pub multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_interval_ms: 50,
            max_interval_ms: 2_000,
            multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Returns true for common transient HTTP status codes.
    #[must_use]
    pub fn is_retryable_status(status: StatusCode) -> bool {
        matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504)
    }

    /// Returns true when `reqwest` reports a connect error or timeout (no response).
    #[must_use]
    pub fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect() || err.is_request()
    }

    /// POST-safe: only transport-level failures (no response body), not generic request build errors.
    #[must_use]
    pub fn is_retryable_post_reqwest_error(err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect()
    }

    /// Status codes that are reasonable to retry for POST (rate limits and gateway overload).
    #[must_use]
    pub fn is_retryable_post_status(status: StatusCode) -> bool {
        matches!(status.as_u16(), 408 | 425 | 429 | 502 | 503 | 504)
    }

    /// Nth backoff delay in milliseconds after attempt `attempt_index` (0 = first retry wait).
    #[must_use]
    pub fn delay_ms_for_attempt(&self, attempt_index: u32) -> u64 {
        let base = self.initial_interval_ms as f64 * self.multiplier.powi(attempt_index as i32);
        let capped = base.min(self.max_interval_ms as f64);
        capped as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_statuses() {
        assert!(RetryPolicy::is_retryable_status(
            StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(RetryPolicy::is_retryable_status(
            StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(!RetryPolicy::is_retryable_status(StatusCode::BAD_REQUEST));
    }

    #[test]
    fn backoff_grows_and_caps() {
        let p = RetryPolicy {
            max_retries: 5,
            initial_interval_ms: 100,
            max_interval_ms: 250,
            multiplier: 2.0,
        };
        assert_eq!(p.delay_ms_for_attempt(0), 100);
        assert_eq!(p.delay_ms_for_attempt(1), 200);
        assert_eq!(p.delay_ms_for_attempt(2), 250);
    }
}
