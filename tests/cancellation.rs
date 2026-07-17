//! Tests for cooperative cancellation via CancellationToken in BatchConfig and ChatOptions.

use std::sync::Arc;

use serde_json::json;
use superglue::batch::{BatchConfig, BatchError, BatchRequest, ErrorStrategy, batch_complete};
use superglue::cancel::CancellationToken;
use superglue::chat::{ChatError, ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn slow_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-slow",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "done" },
            "finish_reason": "stop"
        }]
    })
}

// ---------------------------------------------------------------------------
// CancellationToken basics
// ---------------------------------------------------------------------------

#[test]
fn cancel_token_new_is_not_cancelled() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
}

#[test]
fn cancel_token_child_is_cancelled_when_parent_is() {
    let parent = CancellationToken::new();
    let child = parent.child_token();
    assert!(!child.is_cancelled());
    parent.cancel();
    assert!(child.is_cancelled());
}

#[test]
fn cancel_token_cancelled_is_idempotent() {
    let token = CancellationToken::new();
    token.cancel();
    token.cancel(); // second call should be a no-op
    assert!(token.is_cancelled());
}

// ---------------------------------------------------------------------------
// ChatOptions: pre-cancelled token aborts immediately
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_options_pre_cancelled_returns_cancelled() {
    let server = MockServer::start().await;
    // No mock is registered — if the HTTP call is made the test will get an error, not ChatError::Cancelled.

    let token = CancellationToken::new();
    token.cancel(); // already cancelled before the call

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        cancel: Some(token),
        ..Default::default()
    };

    let err = complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
    )
    .await
    .unwrap_err();

    assert!(
        matches!(err, ChatError::Cancelled),
        "expected ChatError::Cancelled, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// BatchConfig: pre-cancelled token aborts immediately
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_pre_cancelled_returns_cancelled() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(slow_response()))
        .mount(&server)
        .await;

    let token = CancellationToken::new();
    token.cancel(); // already cancelled

    let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };
    let config = BatchConfig {
        max_concurrent: 2,
        error_strategy: ErrorStrategy::Continue,
        cancel: Some(token),
        ..Default::default()
    };
    let requests = vec![BatchRequest::new("hello".to_string())];

    let err = batch_complete(
        http,
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts,
        config,
    )
    .await
    .unwrap_err();

    assert!(
        matches!(err, BatchError::Cancelled),
        "expected BatchError::Cancelled, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// BatchConfig: cancel propagates to per-item options via child token
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batch_without_cancel_completes_normally() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(slow_response()))
        .mount(&server)
        .await;

    let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };
    let config = BatchConfig {
        max_concurrent: 2,
        error_strategy: ErrorStrategy::Continue,
        cancel: None,
        ..Default::default()
    };
    let requests = vec![
        BatchRequest::new("hello".to_string()),
        BatchRequest::new("world".to_string()),
    ];

    let response = batch_complete(
        http,
        Arc::new(ToolRegistry::new()),
        Arc::new(HookRegistry::new()),
        Arc::new(GuardrailRegistry::new()),
        requests,
        &opts,
        config,
    )
    .await
    .unwrap();

    assert_eq!(response.successful, 2);
    assert_eq!(response.failed, 0);
}
