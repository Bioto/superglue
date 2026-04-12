//! Async HTTP client with optional rate limiting, retries on GET, and JSON POST.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
#[cfg(test)]
use reqwest::StatusCode;
use reqwest::{Client, RequestBuilder};
use serde_json::Value;
use tokio::time::sleep;
use tracing::warn;

use crate::http::error::Error;
use crate::http::rate_limit::{DirectRateLimiter, direct_per_second};
use crate::http::retry::RetryPolicy;

/// Configuration for [`HttpClient`].
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Total timeout per attempt (connect + response body for non-streaming).
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub user_agent: String,
    pub retry: RetryPolicy,
    /// When set, every request waits for this per-second quota first.
    pub quota_per_second: Option<std::num::NonZeroU32>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(15),
            user_agent: format!("superglue/{}", env!("CARGO_PKG_VERSION")),
            retry: RetryPolicy::default(),
            quota_per_second: None,
        }
    }
}

/// Thin `reqwest` wrapper with optional [`Governor`](governor)-style QPS cap, GET retries, and JSON POST.
#[derive(Clone)]
pub struct HttpClient {
    inner: Client,
    retry: RetryPolicy,
    limiter: Option<Arc<DirectRateLimiter>>,
}

impl HttpClient {
    /// Build a client from configuration.
    ///
    /// # Errors
    ///
    /// Returns `Error::Reqwest` if the underlying `reqwest::Client` fails to construct.
    pub fn new(config: ClientConfig) -> Result<Self, Error> {
        let inner = Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(config.user_agent)
            .build()?;
        let limiter = config.quota_per_second.map(direct_per_second);
        Ok(Self {
            inner,
            retry: config.retry,
            limiter,
        })
    }

    async fn acquire_limiter(&self) {
        if let Some(lim) = &self.limiter {
            lim.until_ready().await;
        }
    }

    /// GET and buffer the full body. Retries retryable statuses and transport errors per [`RetryPolicy`].
    ///
    /// # Errors
    ///
    /// See [`Error`]. After exhausting retries on a retryable error, returns [`Error::RetriesExhausted`].
    pub async fn get(&self, url: &str) -> Result<Bytes, Error> {
        self.get_with_headers(url, None).await
    }

    /// GET with optional extra headers (e.g. `Accept: text/event-stream`).
    pub async fn get_with_headers(
        &self,
        url: &str,
        headers: Option<&[(&str, &str)]>,
    ) -> Result<Bytes, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_limiter().await;
            let mut req = self.inner.get(url);
            if let Some(h) = headers {
                for (k, v) in h {
                    req = req.header(*k, *v);
                }
            }
            match Self::send_buffered(req).await {
                Ok(b) => return Ok(b),
                Err(e) => {
                    if e.is_retryable() && attempt < self.retry.max_retries {
                        let delay = self.retry.delay_ms_for_attempt(attempt);
                        warn!(attempt, delay_ms = delay, error = %e, "retrying HTTP GET");
                        sleep(Duration::from_millis(delay)).await;
                        attempt += 1;
                        continue;
                    }
                    if e.is_retryable() {
                        return Err(Self::retries_exhausted(&e, attempt));
                    }
                    return Err(e);
                }
            }
        }
    }

    fn retries_exhausted(last: &Error, attempts_used: u32) -> Error {
        let last_status = match last {
            Error::Unsuccessful { status, .. } => Some(*status),
            Error::Reqwest(r) => r.status(),
            _ => None,
        };
        Error::RetriesExhausted {
            attempts: attempts_used.saturating_add(1),
            last: last_status,
        }
    }

    async fn send_buffered(req: RequestBuilder) -> Result<Bytes, Error> {
        let resp = req.send().await?;
        let status = resp.status();
        if status.is_success() {
            Ok(resp.bytes().await?)
        } else {
            let body = resp.bytes().await.unwrap_or_default();
            Err(Error::unsuccessful(status, &body))
        }
    }

    /// GET and stream body chunks. Only the initial `send()` is performed once; non-success status
    /// returns an error before streaming. (Retries after partial body are not attempted.)
    ///
    /// # Errors
    ///
    /// Returns immediately on non-success status with body sample in [`Error::Unsuccessful`].
    pub async fn get_stream(
        &self,
        url: &str,
    ) -> Result<impl Stream<Item = Result<Bytes, Error>> + Send, Error> {
        self.acquire_limiter().await;
        let resp = self.inner.get(url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Err(Error::unsuccessful(status, &body));
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(Error::from)))
    }

    /// `POST` with JSON body. Retries only [`Error::is_retryable_post`] (transport + 429/502/503/504),
    /// not arbitrary 5xx, to reduce duplicate side effects on non-idempotent requests.
    pub async fn post_json(&self, url: &str, body: &Value) -> Result<Value, Error> {
        self.post_json_with_headers(url, body, &[]).await
    }

    /// `POST` with JSON body and extra headers (e.g. `Authorization`, `OpenAI-Organization`).
    pub async fn post_json_with_headers(
        &self,
        url: &str,
        body: &Value,
        headers: &[(&str, &str)],
    ) -> Result<Value, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_limiter().await;
            let mut req = self
                .inner
                .post(url)
                .header("Content-Type", "application/json")
                .json(body);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            match Self::send_json_body(req).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if e.is_retryable_post() && attempt < self.retry.max_retries {
                        let delay = self.retry.delay_ms_for_attempt(attempt);
                        warn!(attempt, delay_ms = delay, error = %e, "retrying HTTP POST JSON");
                        sleep(Duration::from_millis(delay)).await;
                        attempt += 1;
                        continue;
                    }
                    if e.is_retryable_post() {
                        return Err(Self::retries_exhausted(&e, attempt));
                    }
                    return Err(e);
                }
            }
        }
    }

    /// `POST` with JSON body and extra headers, returning the response body as a raw byte stream.
    ///
    /// Use this for `stream: true` requests (Server-Sent Events). No retries are performed after
    /// the connection is established; only the initial `send()` is guarded by the limiter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsuccessful`] immediately on non-2xx status (body preview is buffered).
    pub async fn post_json_stream_with_headers(
        &self,
        url: &str,
        body: &Value,
        headers: &[(&str, &str)],
    ) -> Result<impl Stream<Item = Result<Bytes, Error>> + Send, Error> {
        self.acquire_limiter().await;
        let mut req = self
            .inner
            .post(url)
            .header("Content-Type", "application/json")
            .json(body);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body_bytes = resp.bytes().await.unwrap_or_default();
            return Err(Error::unsuccessful(status, &body_bytes));
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(Error::from)))
    }

    async fn send_json_body(req: RequestBuilder) -> Result<Value, Error> {
        let resp = req.send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(Error::unsuccessful(status, &bytes));
        }
        serde_json::from_slice(&bytes).map_err(|e| Error::InvalidJson(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        let e = Error::Unsuccessful {
            status: StatusCode::SERVICE_UNAVAILABLE,
            preview: String::new(),
            len: 0,
        };
        assert!(e.is_retryable());
    }
}
