//! Model fallback chain integration tests.

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::fallback::{FallbackPolicy, ModelFallbackChain};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient, RetryPolicy};
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;

fn http_no_retry() -> HttpClient {
    let cfg = ClientConfig {
        retry: RetryPolicy {
            max_retries: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    HttpClient::new(cfg).unwrap()
}

fn ok_body(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-ok",
        "model": "backup",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

#[tokio::test]
async fn fallback_on_429_uses_secondary_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("from backup")))
        .mount(&server)
        .await;

    let chain = ModelFallbackChain::new(vec!["primary".into(), "backup".into()]);
    let options = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "primary".into(),
        max_tool_rounds: 4,
        model_fallback: Some(chain),
        ..Default::default()
    };

    let outcome = complete_with_tools(
        &http_no_retry(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &options,
    )
    .await
    .expect("backup should succeed");

    assert_eq!(outcome.content.as_deref(), Some("from backup"));
    assert_eq!(outcome.model_used, "backup");
}

#[tokio::test]
async fn default_policy_does_not_fallback_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    let chain = ModelFallbackChain::new(vec!["primary".into(), "backup".into()]);
    let options = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-bad".to_string()),
        model: "primary".into(),
        max_tool_rounds: 4,
        model_fallback: Some(chain),
        ..Default::default()
    };

    let err = complete_with_tools(
        &http_no_retry(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &options,
    )
    .await
    .expect_err("401 should not fallback");

    assert!(err.to_string().contains("401") || err.to_string().contains("Unauthorized"));
}

#[tokio::test]
async fn custom_policy_fallback_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("ok")))
        .mount(&server)
        .await;

    let chain = ModelFallbackChain::new(vec!["primary".into(), "backup".into()]).with_policy(
        FallbackPolicy {
            retryable_statuses: vec![401, 429, 500, 502, 503, 504],
            on_transport_error: true,
            on_auth_errors: true,
        },
    );
    let options = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "primary".into(),
        max_tool_rounds: 4,
        model_fallback: Some(chain),
        ..Default::default()
    };

    let outcome = complete_with_tools(
        &http_no_retry(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &options,
    )
    .await
    .expect("auth fallback should succeed");

    assert_eq!(outcome.model_used, "backup");
}
