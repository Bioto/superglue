//! Async HTTP client with optional rate limiting, retries on GET, and JSON POST.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
#[cfg(test)]
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use reqwest::{Client, RequestBuilder};
use serde_json::Value;
use tokio::time::sleep;
use tracing::warn;

use crate::http::error::Error;
use crate::http::rate_limit::{DirectRateLimiter, direct_per_second};
use crate::http::retry::RetryPolicy;
use crate::providers::{RateLimitKey, RateLimitRegistry};

/// Configuration for [`HttpClient`].
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Total timeout per attempt (connect + response body for non-streaming).
    pub timeout: Duration,
    /// TCP connection timeout.
    pub connect_timeout: Duration,
    pub user_agent: String,
    pub retry: RetryPolicy,
    /// When set, every request waits for this per-second quota first (legacy global bucket).
    pub quota_per_second: Option<std::num::NonZeroU32>,
    /// Per-API-key rate limits (preferred when set).
    pub rate_limit_registry: Option<Arc<RateLimitRegistry>>,
    /// Maximum number of idle keep-alive connections per host retained in the
    /// pool. Maps directly to `reqwest::ClientBuilder::pool_max_idle_per_host`.
    /// Default: 50 (aligned with gluellm httpx pool keepalive sizing).
    pub pool_max_idle_per_host: usize,
    /// How long an idle connection is kept alive before being evicted from the
    /// pool. `None` uses reqwest's default (90 s). Maps to
    /// `reqwest::ClientBuilder::pool_idle_timeout`.
    pub pool_idle_timeout: Option<Duration>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
            user_agent: format!("superglue/{}", env!("CARGO_PKG_VERSION")),
            retry: RetryPolicy::default(),
            quota_per_second: None,
            rate_limit_registry: None,
            pool_max_idle_per_host: 50,
            pool_idle_timeout: None,
        }
    }
}

impl ClientConfig {
    /// Timeouts suited for LLM streaming (reasoning models may not emit bytes for minutes).
    pub fn for_llm() -> Self {
        Self {
            timeout: Duration::from_secs(600),
            ..Self::default()
        }
    }
}

/// Thin `reqwest` wrapper with optional [`Governor`](governor)-style QPS cap, GET retries, and JSON POST.
#[derive(Clone)]
pub struct HttpClient {
    inner: Client,
    /// Retained so callers can inspect current timeout values and create
    /// derived clients via [`clone_with_timeouts`](Self::clone_with_timeouts).
    pub config: ClientConfig,
    limiter: Option<Arc<DirectRateLimiter>>,
    rate_limit_registry: Option<Arc<RateLimitRegistry>>,
}

impl HttpClient {
    /// Build a client from configuration.
    ///
    /// # Errors
    ///
    /// Returns `Error::Reqwest` if the underlying `reqwest::Client` fails to construct.
    pub fn new(config: ClientConfig) -> Result<Self, Error> {
        let mut builder = Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(config.user_agent.clone())
            .pool_max_idle_per_host(config.pool_max_idle_per_host);
        if let Some(idle_timeout) = config.pool_idle_timeout {
            builder = builder.pool_idle_timeout(idle_timeout);
        }
        let inner = builder.build()?;
        let limiter = config.quota_per_second.map(direct_per_second);
        let rate_limit_registry = config.rate_limit_registry.clone();
        Ok(Self {
            inner,
            config,
            limiter,
            rate_limit_registry,
        })
    }

    /// Return a new `HttpClient` with overridden timeouts, inheriting retry policy and QPS limiter.
    ///
    /// # Errors
    ///
    /// Returns `Error::Reqwest` if the underlying `reqwest::Client` fails to construct.
    pub fn clone_with_timeouts(
        &self,
        timeout: Duration,
        connect_timeout: Duration,
    ) -> Result<Self, Error> {
        let mut new_cfg = self.config.clone();
        new_cfg.timeout = timeout;
        new_cfg.connect_timeout = connect_timeout;
        HttpClient::new(new_cfg)
    }

    async fn acquire_rate_limit(&self, key: Option<RateLimitKey>) {
        if let (Some(reg), Some(k)) = (&self.rate_limit_registry, key) {
            reg.acquire(k).await;
        } else if let Some(lim) = &self.limiter {
            lim.until_ready().await;
        }
    }

    async fn acquire_limiter(&self) {
        self.acquire_rate_limit(None).await;
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
                    if e.is_retryable() && attempt < self.config.retry.max_retries {
                        let delay = retry_delay_ms(&self.config.retry, attempt, &e);
                        warn_http_retry("GET", url, attempt, delay, &e);
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
        let retry_after = parse_retry_after(resp.headers());
        if status.is_success() {
            Ok(resp.bytes().await?)
        } else {
            let body = resp.bytes().await.unwrap_or_default();
            Err(Error::unsuccessful(status, &body, retry_after))
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
        let retry_after = parse_retry_after(resp.headers());
        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Err(Error::unsuccessful(status, &body, retry_after));
        }
        Ok(resp.bytes_stream().map(|r| r.map_err(Error::from)))
    }

    /// `POST` with JSON body. Retries only [`Error::is_retryable_post`] (transport + 429/502/503/504),
    /// not arbitrary 5xx, to reduce duplicate side effects on non-idempotent requests.
    pub async fn post_json(&self, url: &str, body: &Value) -> Result<Value, Error> {
        self.post_json_with_headers(url, body, &[], None).await
    }

    /// `POST` with JSON body and extra headers (e.g. `Authorization`, `OpenAI-Organization`).
    pub async fn post_json_with_headers(
        &self,
        url: &str,
        body: &Value,
        headers: &[(&str, &str)],
        rate_limit_key: Option<RateLimitKey>,
    ) -> Result<Value, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_rate_limit(rate_limit_key).await;
            let mut req = self.inner.post(url);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            req = req.json(body);
            match Self::send_json_body(req).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if e.is_retryable_post() && attempt < self.config.retry.max_retries {
                        let delay = retry_delay_ms(&self.config.retry, attempt, &e);
                        warn_http_retry("POST_JSON", url, attempt, delay, &e);
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
        rate_limit_key: Option<RateLimitKey>,
    ) -> Result<impl Stream<Item = Result<Bytes, Error>> + Send + use<>, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_rate_limit(rate_limit_key).await;
            let mut req = self.inner.post(url);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            req = req.json(body);
            let resp = req.send().await?;
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            if !status.is_success() {
                let body_bytes = resp.bytes().await.unwrap_or_default();
                let err = Error::unsuccessful(status, &body_bytes, retry_after);
                if err.is_retryable_post() && attempt < self.config.retry.max_retries {
                    let delay = retry_delay_ms(&self.config.retry, attempt, &err);
                    warn_http_retry("POST_STREAM", url, attempt, delay, &err);
                    sleep(Duration::from_millis(delay)).await;
                    attempt += 1;
                    continue;
                }
                if err.is_retryable_post() {
                    return Err(Self::retries_exhausted(&err, attempt));
                }
                warn!(
                    status = %status,
                    url = %url,
                    body_len = body_bytes.len(),
                    "POST stream request failed (non-success status)"
                );
                return Err(err);
            }
            return Ok(resp.bytes_stream().map(|r| r.map_err(Error::from)));
        }
    }

    /// `POST` multipart form (e.g. file uploads). Retries like [`Self::post_json_with_headers`].
    pub async fn post_multipart(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        purpose: &str,
        filename: &str,
        file_bytes: Vec<u8>,
        rate_limit_key: Option<RateLimitKey>,
    ) -> Result<Value, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_rate_limit(rate_limit_key).await;
            let part = reqwest::multipart::Part::bytes(file_bytes.clone())
                .file_name(filename.to_string())
                .mime_str("application/octet-stream")
                .map_err(|e| Error::InvalidJson(e.to_string()))?;
            let form = reqwest::multipart::Form::new()
                .text("purpose", purpose.to_string())
                .part("file", part);
            let mut req = self.inner.post(url).multipart(form);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            match Self::send_json_body(req).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if e.is_retryable_post() && attempt < self.config.retry.max_retries {
                        let delay = retry_delay_ms(&self.config.retry, attempt, &e);
                        warn_http_retry("POST_MULTIPART", url, attempt, delay, &e);
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

    async fn send_json_body(req: RequestBuilder) -> Result<Value, Error> {
        let resp = req.send().await?;
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            return Err(Error::unsuccessful(status, &bytes, retry_after));
        }
        serde_json::from_slice(&bytes).map_err(|e| Error::InvalidJson(e.to_string()))
    }
}

/// Parse `Retry-After` as a delay in seconds (common for 429). Caps at 10 minutes. HTTP-date form is ignored.
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers
        .get(reqwest::header::RETRY_AFTER)
        .or_else(|| headers.get("retry-after"))?;
    let s = raw.to_str().ok()?.trim();
    let secs = s.parse::<u64>().ok()?;
    Some(Duration::from_secs(secs.min(600)))
}

/// Extra detail for transport / status errors so logs are not just "error sending request".
fn warn_http_retry(kind: &'static str, url: &str, attempt: u32, delay_ms: u64, err: &Error) {
    match err {
        Error::Reqwest(re) => {
            warn!(
                kind,
                url = %url,
                attempt,
                delay_ms,
                error = %err,
                reqwest_timeout = re.is_timeout(),
                reqwest_connect = re.is_connect(),
                reqwest_decode = re.is_decode(),
                http_status = ?re.status(),
                "retrying HTTP request"
            );
        }
        Error::Unsuccessful {
            status,
            preview,
            len,
            ..
        } => {
            let preview_short: String = preview.chars().take(160).collect();
            warn!(
                kind,
                url = %url,
                attempt,
                delay_ms,
                error = %err,
                http_status = %status,
                response_len = *len,
                response_preview = %preview_short,
                "retrying HTTP request"
            );
        }
        _ => {
            warn!(
                kind,
                url = %url,
                attempt,
                delay_ms,
                error = %err,
                "retrying HTTP request"
            );
        }
    }
}

fn retry_delay_ms(policy: &RetryPolicy, attempt: u32, err: &Error) -> u64 {
    let base = policy.delay_ms_for_attempt(attempt);
    let from_server = err.retry_after_hint().map(|d| d.as_millis().min(600_000) as u64);
    let mut ms = from_server.unwrap_or(base).max(base);
    ms = ms.saturating_add(((attempt as u64).wrapping_mul(31)) % 50);
    ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_client_config_matches_gluellm_pool_sizing() {
        let cfg = ClientConfig::default();
        assert_eq!(cfg.connect_timeout, Duration::from_secs(30));
        assert_eq!(cfg.pool_max_idle_per_host, 50);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
    }

    #[test]
    fn retryable_classification() {
        let e = Error::Unsuccessful {
            status: StatusCode::SERVICE_UNAVAILABLE,
            preview: String::new(),
            len: 0,
            retry_after: None,
        };
        assert!(e.is_retryable());
    }
}
