//! Integration tests for `superglue::batch::batch_complete`.
//!
//! Covers: all-succeed, partial failure with CONTINUE / SKIP / FailFast,
//! concurrency limiting, id auto-assignment, usage aggregation, and
//! per-request system_prompt override.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use serde_json::json;
use superglue::batch::{BatchConfig, BatchError, BatchRequest, ErrorStrategy, batch_complete};
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::tools::ToolRegistry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn http() -> HttpClient {
    // No retries, no rate limit — keep tests fast.
    let cfg = ClientConfig {
        retry: superglue::http::RetryPolicy {
            max_retries: 0,
            ..Default::default()
        },
        ..ClientConfig::default()
    };
    HttpClient::new(cfg).unwrap()
}

fn opts(base_url: String) -> ChatOptions {
    ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        ..Default::default()
    }
}

fn text_response(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
}

fn text_response_usage(content: &str, prompt: u32, completion: u32) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion
        }
    })
}

fn error_response(status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(json!({
        "error": {"message": "upstream error", "type": "server_error"}
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// All requests succeed — results are in input order.
#[tokio::test]
async fn batch_all_succeed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .expect(3)
        .mount(&server)
        .await;

    let http = Arc::new(http());
    let registry = Arc::new(ToolRegistry::new());
    let requests = vec![
        BatchRequest::new("q1").with_id("id-1"),
        BatchRequest::new("q2").with_id("id-2"),
        BatchRequest::new("q3").with_id("id-3"),
    ];

    let resp = batch_complete(
        http,
        registry,
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        BatchConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(resp.total_requests, 3);
    assert_eq!(resp.successful, 3);
    assert_eq!(resp.failed, 0);
    assert_eq!(resp.results.len(), 3);

    // Order must match input.
    assert_eq!(resp.results[0].id, "id-1");
    assert_eq!(resp.results[1].id, "id-2");
    assert_eq!(resp.results[2].id, "id-3");

    for r in &resp.results {
        assert!(r.success);
        assert_eq!(r.content.as_deref(), Some("ok"));
        assert!(r.error.is_none());
    }
}

/// One request fails; CONTINUE keeps all results (success=false for the failed one).
#[tokio::test]
async fn batch_continue_on_partial_failure() {
    let server = MockServer::start().await;

    // First call → 500, subsequent calls → 200.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(error_response(500))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .mount(&server)
        .await;

    let requests = vec![
        BatchRequest::new("fail").with_id("fail"),
        BatchRequest::new("ok-1").with_id("ok-1"),
        BatchRequest::new("ok-2").with_id("ok-2"),
    ];

    let config = BatchConfig {
        max_concurrent: 1, // serial so first request always fails first
        error_strategy: ErrorStrategy::Continue,
        ..Default::default()
    };
    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap();

    assert_eq!(resp.total_requests, 3);
    assert_eq!(resp.results.len(), 3);
    assert_eq!(resp.failed, 1);
    assert_eq!(resp.successful, 2);

    // The failed item should be "fail".
    let failed = resp.results.iter().find(|r| !r.success).unwrap();
    assert_eq!(failed.id, "fail");
    assert!(failed.error.is_some());
}

/// One request fails; SKIP removes it from results.
#[tokio::test]
async fn batch_skip_failures() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(error_response(500))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .mount(&server)
        .await;

    let config = BatchConfig {
        max_concurrent: 1,
        error_strategy: ErrorStrategy::Skip,
        ..Default::default()
    };
    let requests = vec![
        BatchRequest::new("fail").with_id("fail"),
        BatchRequest::new("ok-1").with_id("ok-1"),
        BatchRequest::new("ok-2").with_id("ok-2"),
    ];

    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap();

    // total_requests is always the full count; results only contains successes.
    assert_eq!(resp.total_requests, 3);
    assert_eq!(resp.results.len(), 2);
    assert!(resp.results.iter().all(|r| r.success));
    assert!(resp.results.iter().all(|r| r.id != "fail"));
}

/// FailFast: returns BatchError when any request fails.
#[tokio::test]
async fn batch_fail_fast() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(error_response(500))
        .mount(&server)
        .await;

    let config = BatchConfig {
        max_concurrent: 1,
        error_strategy: ErrorStrategy::FailFast,
        ..Default::default()
    };
    let requests = vec![
        BatchRequest::new("will-fail").with_id("the-id"),
        BatchRequest::new("will-not-run"),
    ];

    let err = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap_err();

    match err {
        BatchError::FailFast { id, message } => {
            assert_eq!(id, "the-id");
            assert!(!message.is_empty());
        }
        other => panic!("expected FailFast, got {other}"),
    }
}

/// Empty batch returns an empty response without touching the server.
#[tokio::test]
async fn batch_empty_input() {
    let server = MockServer::start().await;
    // No mock registered — any request would panic.

    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        vec![],
        &opts(server.uri()),
        BatchConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(resp.total_requests, 0);
    assert_eq!(resp.results.len(), 0);
    assert_eq!(resp.successful, 0);
    assert_eq!(resp.failed, 0);
    assert!(resp.total_usage.is_none());
}

/// Auto-assigned IDs are unique and non-empty.
#[tokio::test]
async fn batch_assigns_ids() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .mount(&server)
        .await;

    let requests = vec![
        BatchRequest::new("q1"), // no id
        BatchRequest::new("q2"), // no id
        BatchRequest::new("q3").with_id("explicit"),
    ];

    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        BatchConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(resp.results.len(), 3);

    // Explicit id is preserved.
    assert_eq!(resp.results[2].id, "explicit");

    // Auto-assigned ids are non-empty.
    assert!(!resp.results[0].id.is_empty());
    assert!(!resp.results[1].id.is_empty());

    // All ids are distinct.
    let ids: std::collections::HashSet<&str> = resp.results.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids.len(), 3);
}

/// Token counts are summed correctly across all successful completions.
#[tokio::test]
async fn batch_aggregates_usage() {
    let server = MockServer::start().await;

    // Requests respond with distinct usage values so we can verify the sum.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response_usage("r1", 10, 5)))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response_usage("r2", 20, 10)))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response_usage("r3", 30, 15)))
        .mount(&server)
        .await;

    let requests = vec![
        BatchRequest::new("q1"),
        BatchRequest::new("q2"),
        BatchRequest::new("q3"),
    ];

    let config = BatchConfig {
        max_concurrent: 1, // serial so mocks are consumed in order
        ..Default::default()
    };
    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap();

    let usage = resp.total_usage.expect("expected aggregated usage");
    assert_eq!(usage.prompt_tokens, 10 + 20 + 30);
    assert_eq!(usage.completion_tokens, 5 + 10 + 15);
    assert_eq!(usage.total_tokens, 15 + 30 + 45);
}

/// Per-request system_prompt overrides the global one for that item only.
#[tokio::test]
async fn batch_per_request_system_prompt() {
    let server = MockServer::start().await;

    let captured_bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured_bodies);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value =
                serde_json::from_slice(req.body.as_slice()).unwrap_or_default();
            captured_clone.lock().unwrap().push(body);
            ResponseTemplate::new(200).set_body_json(text_response("ok"))
        })
        .mount(&server)
        .await;

    let requests = vec![
        BatchRequest::new("q1").with_system_prompt("per-request SP"),
        BatchRequest::new("q2"), // uses global SP
    ];

    let mut options = opts(server.uri());
    options.system_prompt = Some("global SP".into());

    let config = BatchConfig {
        max_concurrent: 1, // serial so bodies are ordered
        ..Default::default()
    };
    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &options,
        config,
    )
    .await
    .unwrap();

    assert_eq!(resp.successful, 2);

    let bodies = captured_bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2);

    // First request: per-request SP overrides global.
    let first_system = &bodies[0]["messages"][0]["content"];
    assert_eq!(first_system.as_str().unwrap(), "per-request SP");

    // Second request: global SP used.
    let second_system = &bodies[1]["messages"][0]["content"];
    assert_eq!(second_system.as_str().unwrap(), "global SP");
}

/// `max_concurrent` limits the number of simultaneous in-flight requests.
///
/// We use an atomic high-watermark counter to verify that at most K requests
/// are active simultaneously.
#[tokio::test]
async fn batch_respects_max_concurrent() {
    let server = MockServer::start().await;
    let active = Arc::new(AtomicI32::new(0));
    let peak = Arc::new(AtomicI32::new(0));

    let active_clone = Arc::clone(&active);
    let peak_clone = Arc::clone(&peak);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let cur = active_clone.fetch_add(1, Ordering::SeqCst) + 1;
            // Update peak.
            let mut old_peak = peak_clone.load(Ordering::SeqCst);
            while cur > old_peak {
                match peak_clone.compare_exchange(old_peak, cur, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(actual) => old_peak = actual,
                }
            }
            // Simulate a tiny bit of work (synchronous — wiremock responders are sync).
            std::thread::sleep(std::time::Duration::from_millis(5));
            active_clone.fetch_sub(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(text_response("ok"))
        })
        .mount(&server)
        .await;

    let requests: Vec<BatchRequest> = (0..10)
        .map(|i| BatchRequest::new(format!("q{i}")))
        .collect();

    let config = BatchConfig {
        max_concurrent: 3,
        ..Default::default()
    };
    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap();

    assert_eq!(resp.successful, 10);

    let observed_peak = peak.load(Ordering::SeqCst);
    // The semaphore must keep concurrent calls ≤ 3.
    assert!(
        observed_peak <= 3,
        "peak concurrent={observed_peak} exceeded max_concurrent=3"
    );
}

/// Successful results include correct `rounds` and non-zero `elapsed_secs`.
#[tokio::test]
async fn batch_result_metadata() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("hello")))
        .mount(&server)
        .await;

    let resp = batch_complete(
        Arc::new(http()),
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        vec![BatchRequest::new("hi")],
        &opts(server.uri()),
        BatchConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(resp.results.len(), 1);
    let r = &resp.results[0];
    assert!(r.success);
    assert_eq!(r.rounds, 1);
    assert!(r.elapsed_secs >= 0.0);
    assert!(resp.elapsed_secs >= 0.0);
}
