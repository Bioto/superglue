//! Rate limiting tests.
//!
//! The `quota_per_second` field on [`ClientConfig`] installs a governor
//! token-bucket limiter. These tests verify that the limiter:
//!   1. Delays requests so the measured throughput does not exceed the cap.
//!   2. Does not add unnecessary delay when no cap is configured.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use serde_json::json;
use superglue::http::{ClientConfig, HttpClient};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// QPS cap enforced
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limiter_caps_throughput() {
    // Cap at 2 requests/second. Sending 3 requests means at least 1 second
    // must elapse (the governor won't issue token #3 until t≈1s).
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.quota_per_second = Some(NonZeroU32::new(2).unwrap());
    let http = HttpClient::new(cfg).unwrap();

    let url = server.uri();
    let start = Instant::now();
    for _ in 0..3 {
        http.get(&url).await.unwrap();
    }
    let elapsed = start.elapsed();

    // With a 2 req/s limiter:
    // - token 1 available immediately
    // - token 2 available immediately (burst = 1 for governor default)
    // - token 3 available at ~500ms or ~1s depending on burst setting
    // We assert at least 400ms to give some slack.
    assert!(
        elapsed.as_millis() >= 400,
        "expected ≥400ms with 2 req/s cap, got {elapsed:?}"
    );
}

#[tokio::test]
async fn rate_limiter_post_caps_throughput() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.quota_per_second = Some(NonZeroU32::new(2).unwrap());
    let http = HttpClient::new(cfg).unwrap();

    let url = server.uri();
    let start = Instant::now();
    for _ in 0..3 {
        http.post_json_with_headers(&url, &json!({}), &[])
            .await
            .unwrap();
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() >= 400,
        "expected ≥400ms with 2 req/s cap on POST, got {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// No cap — requests complete without artificial delay
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_rate_limiter_is_fast() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let url = server.uri();
    let start = Instant::now();
    for _ in 0..10 {
        http.get(&url).await.unwrap();
    }
    let elapsed = start.elapsed();

    // 10 local loopback requests with no throttle should be well under 1s.
    assert!(
        elapsed.as_millis() < 1000,
        "10 unthrottled requests took unexpectedly long: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// All requests still succeed under the cap
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limited_requests_all_succeed() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("GET"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_string("ok")
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.quota_per_second = Some(NonZeroU32::new(5).unwrap());
    let http = HttpClient::new(cfg).unwrap();

    let url = server.uri();
    for _ in 0..5 {
        http.get(&url).await.unwrap();
    }
    assert_eq!(count.load(Ordering::SeqCst), 5);
}

// ---------------------------------------------------------------------------
// Limiter applies per client instance (two independent clients)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_clients_independent_limits() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .mount(&server)
        .await;

    let mut cfg1 = ClientConfig::default();
    cfg1.quota_per_second = Some(NonZeroU32::new(2).unwrap());

    let cfg2 = ClientConfig::default(); // uncapped

    let capped = HttpClient::new(cfg1).unwrap();
    let uncapped = HttpClient::new(cfg2).unwrap();

    let url = server.uri();

    // Uncapped client does 5 requests fast
    let start = Instant::now();
    for _ in 0..5 {
        uncapped.get(&url).await.unwrap();
    }
    let uncapped_ms = start.elapsed().as_millis();

    // Capped client at 2/s should be slower for the same 5 requests
    let start = Instant::now();
    for _ in 0..5 {
        capped.get(&url).await.unwrap();
    }
    let capped_ms = start.elapsed().as_millis();

    assert!(
        capped_ms > uncapped_ms,
        "capped client ({capped_ms}ms) should be slower than uncapped ({uncapped_ms}ms)"
    );
}
