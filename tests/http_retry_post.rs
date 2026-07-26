//! POST-specific retry behaviour.
//!
//! POST retries are intentionally narrower than GET retries to reduce
//! duplicate side-effects: only transport errors + 429/502/503/504 are
//! retried. All other 4xx and 5xx are returned immediately.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;
use superglue::http::{ClientConfig, HttpClient, RetryPolicy};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fast_retry(max: u32) -> RetryPolicy {
    RetryPolicy {
        max_retries: max,
        initial_interval_ms: 1,
        max_interval_ms: 5,
        multiplier: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Statuses that SHOULD be retried on POST
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_retries_429() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(429).set_body_string("rate limited")
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"ok": true}))
            }
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(2);
    let http = HttpClient::new(cfg).unwrap();
    let result = http
        .post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn post_retries_502() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(502).set_body_string("bad gateway")
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"ok": true}))
            }
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(2);
    let http = HttpClient::new(cfg).unwrap();
    assert!(
        http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
            .await
            .is_ok()
    );
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn post_retries_503() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                ResponseTemplate::new(503)
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"ok": true}))
            }
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    assert!(
        http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
            .await
            .is_ok()
    );
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn post_retries_504() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(504)
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"ok": true}))
            }
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(2);
    let http = HttpClient::new(cfg).unwrap();
    assert!(
        http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
            .await
            .is_ok()
    );
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// Statuses that must NOT be retried on POST
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_does_not_retry_400() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(400).set_body_string("bad request")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    let err = http
        .post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    // Should fail immediately with 400, not retry
    assert_eq!(count.load(Ordering::SeqCst), 1, "should not retry 400");
    assert!(err.to_string().contains("400") || err.to_string().contains("Unsuccessful"));
}

#[tokio::test]
async fn post_does_not_retry_401() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(401).set_body_string("unauthorized")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    assert_eq!(count.load(Ordering::SeqCst), 1, "should not retry 401");
}

#[tokio::test]
async fn post_does_not_retry_403() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(403).set_body_string("forbidden")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    assert_eq!(count.load(Ordering::SeqCst), 1, "should not retry 403");
}

#[tokio::test]
async fn post_does_not_retry_500() {
    // 500 Internal Server Error is ambiguous for POST idempotency — we do NOT
    // retry it to avoid duplicate side-effects.
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(500).set_body_string("internal error")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    assert_eq!(count.load(Ordering::SeqCst), 1, "should not retry 500");
}

// ---------------------------------------------------------------------------
// Retry exhaustion
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_exhausts_retries_on_persistent_429() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(429).set_body_string("always rate limited")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(3);
    let http = HttpClient::new(cfg).unwrap();
    let err = http
        .post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    // 1 original attempt + 3 retries = 4 total
    assert_eq!(count.load(Ordering::SeqCst), 4);
    let msg = err.to_string();
    assert!(
        msg.contains("retry") || msg.contains("Retries") || msg.contains("429"),
        "expected retry-exhausted error, got: {msg}"
    );
}

#[tokio::test]
async fn post_exhausts_retries_on_persistent_503() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(503)
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = fast_retry(2);
    let http = HttpClient::new(cfg).unwrap();
    http.post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    assert_eq!(count.load(Ordering::SeqCst), 3); // 1 + 2 retries
}

// ---------------------------------------------------------------------------
// Response body parsing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_returns_json_body_on_200() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"answer": 42})))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let val = http
        .post_json_with_headers(&server.uri(), &json!({"q": "?"}), &[], None)
        .await
        .unwrap();
    assert_eq!(val["answer"], 42);
}

#[tokio::test]
async fn post_returns_error_on_non_json_200() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .respond_with(ResponseTemplate::new(200).set_body_string("this is not json"))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let err = http
        .post_json_with_headers(&server.uri(), &json!({}), &[], None)
        .await
        .unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("json"),
        "expected JSON parse error, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Custom headers forwarded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn post_sends_custom_headers() {
    use wiremock::matchers::header;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .and(header("X-Custom-Key", "secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let result = http
        .post_json_with_headers(
            &server.uri(),
            &json!({}),
            &[("X-Custom-Key", "secret")],
            None,
        )
        .await;
    assert!(result.is_ok(), "custom header not forwarded: {result:?}");
}
