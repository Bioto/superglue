//! Integration tests against a local mock HTTP server.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use futures_util::StreamExt;
use superglue::http::{ClientConfig, HttpClient, RetryPolicy, SseParser};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn get_succeeds_on_200() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
        .mount(&server)
        .await;

    let client = HttpClient::new(ClientConfig::default()).unwrap();
    let body = client.get(&server.uri()).await.unwrap();
    assert_eq!(body.as_ref(), b"hello");
}

#[tokio::test]
async fn get_retries_then_succeeds() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("GET"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                ResponseTemplate::new(503).set_body_string("unavailable")
            } else {
                ResponseTemplate::new(200).set_body_string("ok")
            }
        })
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = RetryPolicy {
        max_retries: 3,
        initial_interval_ms: 1,
        max_interval_ms: 10,
        multiplier: 2.0,
    };
    let client = HttpClient::new(cfg).unwrap();
    let body = client.get(&server.uri()).await.unwrap();
    assert_eq!(body.as_ref(), b"ok");
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn get_exhausts_retries_on_persistent_503() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503).set_body_string("no"))
        .mount(&server)
        .await;

    let mut cfg = ClientConfig::default();
    cfg.retry = RetryPolicy {
        max_retries: 2,
        initial_interval_ms: 1,
        max_interval_ms: 5,
        multiplier: 2.0,
    };
    let client = HttpClient::new(cfg).unwrap();
    let err = client.get(&server.uri()).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("retry") || msg.contains("503") || msg.contains("Retries"),
        "{msg}"
    );
}

#[tokio::test]
async fn stream_collects_sse_events() {
    let server = MockServer::start().await;
    let body = "data: one\n\ndata: two\n\n";
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = HttpClient::new(ClientConfig::default()).unwrap();
    let url = server.uri();
    let mut stream = client.get_stream(&url).await.unwrap();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        buf.push_str(&String::from_utf8_lossy(&chunk));
    }
    let mut parser = SseParser::new();
    let events = parser.push_str(&buf).unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].data, "one");
    assert_eq!(events[1].data, "two");
}
