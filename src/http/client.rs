//! Async HTTP client with optional rate limiting, retries on GET, and JSON POST.

use std::time::{Duration, Instant};
use std::{pin::Pin, sync::Arc};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
#[cfg(test)]
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use reqwest::{Client, RequestBuilder};
use serde_json::Value;
use tokio::time::sleep;
use tracing::warn;

use crate::http::error::{Error, StreamTimeoutPhase};
use crate::http::rate_limit::{DirectRateLimiter, direct_per_second};
use crate::http::retry::RetryPolicy;
use crate::providers::{RateLimitKey, RateLimitRegistry};

/// Configuration for [`HttpClient`].
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Total timeout per attempt for non-streaming requests.
    pub timeout: Duration,
    /// TCP connection timeout.
    pub connect_timeout: Duration,
    /// Maximum time to wait for the response headers and first body chunk.
    pub stream_first_byte_timeout: Duration,
    /// Maximum time between body chunks after a stream produces its first chunk.
    pub stream_idle_timeout: Duration,
    /// Hard deadline for reading a non-2xx error response body. Kept short so a
    /// server that sends error headers then stalls can't hold the connection for
    /// the full LLM `timeout` (which can be 600s).
    pub error_body_timeout: Duration,
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
            stream_first_byte_timeout: Duration::from_secs(60),
            stream_idle_timeout: Duration::from_secs(60),
            error_body_timeout: Duration::from_secs(5),
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
    ///
    /// Attaches the process-wide rate-limit registry so TypeSafe calls share one QPS bucket.
    pub fn for_llm() -> Self {
        Self {
            timeout: Duration::from_secs(600),
            stream_first_byte_timeout: Duration::from_secs(600),
            stream_idle_timeout: Duration::from_secs(180),
            rate_limit_registry: Some(RateLimitRegistry::shared()),
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

    /// Return a new `HttpClient` with overridden request timeouts.
    ///
    /// The underlying `reqwest::Client` is cloned, so the connection pool remains shared.
    /// `connect_timeout` is retained in the derived configuration, but reqwest applies the
    /// connection timeout when the shared client is built.
    ///
    /// The `Result` return type is retained for API compatibility. Cloning the client is infallible.
    pub fn clone_with_timeouts(
        &self,
        timeout: Duration,
        connect_timeout: Duration,
    ) -> Result<Self, Error> {
        let mut config = self.config.clone();
        config.timeout = timeout;
        config.connect_timeout = connect_timeout;
        Ok(Self {
            inner: self.inner.clone(),
            config,
            limiter: self.limiter.clone(),
            rate_limit_registry: self.rate_limit_registry.clone(),
        })
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
            let mut req = self.inner.get(url).timeout(self.config.timeout);
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
                        return Err(Self::retries_exhausted("GET", url, e, attempt));
                    }
                    return Err(e);
                }
            }
        }
    }

    fn retries_exhausted(kind: &'static str, url: &str, last: Error, attempts_used: u32) -> Error {
        let attempts = attempts_used.saturating_add(1);
        let status = last.status();
        let preview = last.provider_message().unwrap_or_else(|| last.to_string());
        warn!(
            kind,
            host = %url_host(url),
            attempts,
            http_status = ?status,
            response_preview = %preview,
            "HTTP retry budget exhausted"
        );
        Error::RetriesExhausted {
            attempts,
            last: Box::new(last),
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

    /// Read an error-response body under a hard deadline. A server that sends
    /// non-2xx headers then stalls must not be able to hang a streaming call,
    /// since per-attempt reqwest timeouts no longer cover stream paths.
    async fn read_error_body(resp: reqwest::Response, timeout: Duration) -> (Bytes, bool) {
        match tokio::time::timeout(timeout, resp.bytes()).await {
            Ok(Ok(body)) => (body, false),
            Ok(Err(_)) | Err(_) => (Bytes::new(), true),
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
    ) -> Result<impl Stream<Item = Result<Bytes, Error>> + Send + use<>, Error> {
        self.acquire_limiter().await;
        let first_byte_deadline = Instant::now() + self.config.stream_first_byte_timeout;
        let resp = tokio::time::timeout(
            self.config.stream_first_byte_timeout,
            self.inner.get(url).send(),
        )
        .await
        .map_err(|_| Error::StreamTimeout {
            phase: StreamTimeoutPhase::FirstByte,
        })??;
        let status = resp.status();
        let retry_after = parse_retry_after(resp.headers());
        if !status.is_success() {
            let (body, body_read_timed_out) =
                Self::read_error_body(resp, self.config.error_body_timeout).await;
            if body_read_timed_out {
                warn!(
                    status = %status,
                    host = %url_host(url),
                    "GET stream error response body read timed out"
                );
            }
            return Err(Error::unsuccessful(status, &body, retry_after));
        }
        Ok(timeout_stream(
            resp.bytes_stream(),
            first_byte_deadline,
            self.config.stream_idle_timeout,
        ))
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
            let mut req = self.inner.post(url).timeout(self.config.timeout);
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
                        return Err(Self::retries_exhausted("POST_JSON", url, e, attempt));
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
            let first_byte_deadline = Instant::now() + self.config.stream_first_byte_timeout;
            let mut req = self.inner.post(url);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            req = req.json(body);
            let resp = tokio::time::timeout(self.config.stream_first_byte_timeout, req.send())
                .await
                .map_err(|_| Error::StreamTimeout {
                    phase: StreamTimeoutPhase::FirstByte,
                })??;
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            if !status.is_success() {
                let (body_bytes, body_read_timed_out) =
                    Self::read_error_body(resp, self.config.error_body_timeout).await;
                let err = Error::unsuccessful(status, &body_bytes, retry_after);
                if err.is_retryable_post() && attempt < self.config.retry.max_retries {
                    let delay = retry_delay_ms(&self.config.retry, attempt, &err);
                    warn_http_retry("POST_STREAM", url, attempt, delay, &err);
                    sleep(Duration::from_millis(delay)).await;
                    attempt += 1;
                    continue;
                }
                if err.is_retryable_post() {
                    return Err(Self::retries_exhausted("POST_STREAM", url, err, attempt));
                }
                warn!(
                    status = %status,
                    host = %url_host(url),
                    body_len = body_bytes.len(),
                    body_read_timed_out,
                    "POST stream request failed (non-success status)"
                );
                return Err(err);
            }
            return Ok(timeout_stream(
                resp.bytes_stream(),
                first_byte_deadline,
                self.config.stream_idle_timeout,
            ));
        }
    }

    /// `POST` multipart form with extra text fields plus one file part.
    pub async fn post_multipart_with_fields(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        fields: &[(&str, &str)],
        filename: &str,
        file_bytes: Vec<u8>,
        mime: &str,
        rate_limit_key: Option<RateLimitKey>,
    ) -> Result<Value, Error> {
        let mut attempt: u32 = 0;
        loop {
            self.acquire_rate_limit(rate_limit_key).await;
            let part = reqwest::multipart::Part::bytes(file_bytes.clone())
                .file_name(filename.to_string())
                .mime_str(mime)
                .map_err(|e| Error::InvalidJson(e.to_string()))?;
            let mut form = reqwest::multipart::Form::new();
            for (name, value) in fields {
                form = form.text((*name).to_string(), (*value).to_string());
            }
            form = form.part("file", part);
            let mut req = self
                .inner
                .post(url)
                .timeout(self.config.timeout)
                .multipart(form);
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
                        return Err(Self::retries_exhausted("POST_MULTIPART", url, e, attempt));
                    }
                    return Err(e);
                }
            }
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
            let mut req = self
                .inner
                .post(url)
                .timeout(self.config.timeout)
                .multipart(form);
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
                        return Err(Self::retries_exhausted("POST_MULTIPART", url, e, attempt));
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

/// Apply first-byte and inter-chunk deadlines without imposing a total stream lifetime.
fn timeout_stream<S>(
    stream: S,
    first_byte_deadline: Instant,
    idle_timeout: Duration,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = Box::pin(stream);
    Box::pin(futures_util::stream::unfold(
        Some((stream, true)),
        move |state| async move {
            let (mut stream, first) = state?;
            let limit = if first {
                first_byte_deadline.saturating_duration_since(Instant::now())
            } else {
                idle_timeout
            };
            match tokio::time::timeout(limit, stream.next()).await {
                Ok(Some(Ok(bytes))) => Some((Ok(bytes), Some((stream, false)))),
                Ok(Some(Err(error))) => Some((Err(error.into()), None)),
                Ok(None) => None,
                Err(_) => Some((
                    Err(Error::StreamTimeout {
                        phase: if first {
                            StreamTimeoutPhase::FirstByte
                        } else {
                            StreamTimeoutPhase::Idle
                        },
                    }),
                    None,
                )),
            }
        },
    ))
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

fn url_host(url: &str) -> String {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
        .unwrap_or_else(|| "[invalid-url]".to_string())
}

/// Extra detail for transport / status errors so logs are not just "error sending request".
fn warn_http_retry(kind: &'static str, url: &str, attempt: u32, delay_ms: u64, err: &Error) {
    match err {
        Error::Reqwest(re) => {
            warn!(
                kind,
                host = %url_host(url),
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
                host = %url_host(url),
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
                host = %url_host(url),
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
    let from_server = err
        .retry_after_hint()
        .map(|d| d.as_millis().min(600_000) as u64);
    let mut ms = from_server.unwrap_or(base).max(base);
    ms = ms.saturating_add(((attempt as u64).wrapping_mul(31)) % 50);
    ms
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::StreamTimeoutPhase;
    use futures_util::stream;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn default_client_config_matches_gluellm_pool_sizing() {
        let cfg = ClientConfig::default();
        assert_eq!(cfg.connect_timeout, Duration::from_secs(30));
        assert_eq!(cfg.pool_max_idle_per_host, 50);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
        assert_eq!(cfg.stream_first_byte_timeout, Duration::from_secs(60));
        assert_eq!(cfg.stream_idle_timeout, Duration::from_secs(60));
    }

    #[test]
    fn llm_client_config_separates_stream_timeouts() {
        let cfg = ClientConfig::for_llm();
        assert_eq!(cfg.timeout, Duration::from_secs(600));
        assert_eq!(cfg.stream_first_byte_timeout, Duration::from_secs(600));
        assert_eq!(cfg.stream_idle_timeout, Duration::from_secs(180));
        let registry = cfg.rate_limit_registry.expect("shared rate limit registry");
        let key = RateLimitKey {
            provider: crate::providers::ProviderId::TypeSafe,
            key_id: crate::providers::api_key_id(&secrecy::SecretString::from("ts-llm")),
        };
        assert_eq!(
            registry.qps_for(key).get(),
            crate::providers::TYPESAFE_DEFAULT_QPS
        );
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
        let timeout = Error::StreamTimeout {
            phase: StreamTimeoutPhase::Idle,
        };
        assert!(timeout.is_retryable());
        assert!(timeout.is_retryable_post());
        assert!(timeout.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn stream_can_live_beyond_first_byte_deadline() {
        let source = stream::unfold(0, |index| async move {
            if index == 8 {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(8)).await;
            Some((Ok(Bytes::from_static(b"chunk")), index + 1))
        });
        let first_deadline = Instant::now() + Duration::from_millis(30);
        let mut stream = timeout_stream(source, first_deadline, Duration::from_millis(20));
        let mut count = 0;
        while let Some(chunk) = stream.next().await {
            assert!(chunk.is_ok());
            count += 1;
        }
        assert_eq!(count, 8);
    }

    #[tokio::test]
    async fn stream_times_out_before_first_byte() {
        let source = stream::once(async {
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(Bytes::from_static(b"late"))
        });
        let mut stream = timeout_stream(
            source,
            Instant::now() + Duration::from_millis(5),
            Duration::from_secs(1),
        );
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            Error::StreamTimeout {
                phase: StreamTimeoutPhase::FirstByte
            }
        ));
    }

    #[tokio::test]
    async fn stream_times_out_after_an_idle_gap() {
        let source = stream::unfold(0, |index| async move {
            if index == 0 {
                return Some((Ok(Bytes::from_static(b"first")), 1));
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
            Some((Ok(Bytes::from_static(b"late")), 2))
        });
        let mut stream = timeout_stream(
            source,
            Instant::now() + Duration::from_secs(1),
            Duration::from_millis(5),
        );
        assert!(stream.next().await.unwrap().is_ok());
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            Error::StreamTimeout {
                phase: StreamTimeoutPhase::Idle
            }
        ));
    }

    #[tokio::test]
    async fn stream_heartbeat_bytes_reset_idle_timeout() {
        let source = stream::unfold(0, |index| async move {
            if index == 5 {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(3)).await;
            let bytes = if index == 4 {
                Bytes::from_static(b"data: done\n\n")
            } else {
                Bytes::from_static(b": ping\n\n")
            };
            Some((Ok(bytes), index + 1))
        });
        let mut stream = timeout_stream(
            source,
            Instant::now() + Duration::from_secs(1),
            Duration::from_millis(10),
        );
        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk.unwrap());
        }
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks.last(), Some(&Bytes::from_static(b"data: done\n\n")));
    }

    #[tokio::test]
    async fn http_stream_can_outlive_non_stream_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for event in [
                &b"data: one\n\n"[..],
                &b"data: two\n\n"[..],
                &b"data: three\n\n"[..],
            ] {
                socket.write_all(event).await.unwrap();
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        });
        let config = ClientConfig {
            timeout: Duration::from_millis(5),
            connect_timeout: Duration::from_secs(1),
            stream_first_byte_timeout: Duration::from_secs(1),
            stream_idle_timeout: Duration::from_millis(100),
            ..ClientConfig::default()
        };
        let client = HttpClient::new(config).unwrap();
        let url = format!("http://{address}/events");
        let mut response = client.get_stream(&url).await.unwrap();
        let mut body = Vec::new();
        while let Some(chunk) = response.next().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert!(body.starts_with(b"data: one"));
        assert!(body.windows(5).any(|window| window == b"three"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stream_error_body_read_is_bounded() {
        // Server sends non-2xx headers then never writes the body. Without the
        // bound on read_error_body this would hang forever; with `timeout: 100ms`
        // the buffered body read must give up and return Unsuccessful quickly.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 999\r\n\r\n")
                .await
                .unwrap();
            // Hold the socket open without sending the body.
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let config = ClientConfig {
            timeout: Duration::from_millis(100),
            connect_timeout: Duration::from_secs(1),
            stream_first_byte_timeout: Duration::from_secs(1),
            stream_idle_timeout: Duration::from_secs(1),
            error_body_timeout: Duration::from_millis(100),
            retry: RetryPolicy {
                max_retries: 0,
                ..RetryPolicy::default()
            },
            ..ClientConfig::default()
        };
        let client = HttpClient::new(config).unwrap();
        let url = format!("http://{address}/boom");
        let started = Instant::now();
        let err = match client.get_stream(&url).await {
            Err(e) => e,
            Ok(_) => panic!("expected error response, got a stream"),
        };
        assert!(
            matches!(
                err,
                Error::Unsuccessful {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    ..
                }
            ),
            "expected Unsuccessful(500), got {err:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "error-body read was not bounded: elapsed {:?}",
            started.elapsed()
        );
        server.abort();
    }

    #[tokio::test]
    async fn stream_post_retry_gets_fresh_first_byte_deadline() {
        // First request: 429 with Retry-After: 0 (no meaningful backoff). The
        // retry must get a fresh first-byte deadline computed after the limiter
        // and backoff — a stale deadline would fail the retry immediately.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            // First request: 429, keep-alive (no Connection: close).
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            // Second (retry) request on the same socket: valid SSE stream.
            let mut request2 = [0_u8; 4096];
            let _ = socket.read(&mut request2).await.unwrap();
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: ok\n\n")
                .await
                .unwrap();
        });
        let config = ClientConfig {
            timeout: Duration::from_millis(500),
            connect_timeout: Duration::from_secs(1),
            stream_first_byte_timeout: Duration::from_millis(500),
            stream_idle_timeout: Duration::from_secs(1),
            retry: RetryPolicy {
                max_retries: 2,
                initial_interval_ms: 1,
                max_interval_ms: 1,
                multiplier: 1.0,
            },
            ..ClientConfig::default()
        };
        let client = HttpClient::new(config).unwrap();
        let url = format!("http://{address}/retry");
        let mut response = client
            .post_json_stream_with_headers(&url, &serde_json::json!({}), &[], None)
            .await
            .unwrap();
        let mut body = Vec::new();
        while let Some(chunk) = response.next().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(&body, b"data: ok\n\n");
        server.abort();
    }
}
