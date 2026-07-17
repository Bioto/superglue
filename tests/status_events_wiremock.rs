//! Integration tests for StatusEmitter process events.

use std::sync::Arc;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::events::{CollectingSubscriber, ProcessEventKind, StatusEmitter, StatusSubscriber};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;

fn http() -> HttpClient {
    let cfg = ClientConfig {
        retry: superglue::http::RetryPolicy {
            max_retries: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    HttpClient::new(cfg).unwrap()
}

fn opts(base_url: String, emitter: Arc<StatusEmitter>) -> ChatOptions {
    ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "gpt-5.4-nano-2026-03-17-mini".into(),
        max_tool_rounds: 4,
        status_emitter: Some(emitter),
        ..Default::default()
    }
}

#[tokio::test]
async fn completion_emits_llm_call_end_with_cost() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-5.4-nano-2026-03-17-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
        })))
        .mount(&server)
        .await;

    let emitter = Arc::new(StatusEmitter::new());
    let collector = Arc::new(CollectingSubscriber::new());
    emitter
        .subscribe(Arc::clone(&collector) as Arc<dyn StatusSubscriber>)
        .await;

    let options = opts(server.uri(), Arc::clone(&emitter));
    let outcome = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &options,
    )
    .await
    .expect("completion should succeed");

    assert_eq!(outcome.content.as_deref(), Some("hello"));

    let events = collector.take_events().await;
    assert!(events.iter().any(|e| e.kind == ProcessEventKind::LlmCallStart));
    let end = events
        .iter()
        .find(|e| e.kind == ProcessEventKind::LlmCallEnd)
        .expect("llm_call_end");
    assert!(end.estimated_cost_usd.unwrap_or(0.0) > 0.0);
    assert_eq!(end.usage.as_ref().map(|u| u.total_tokens), Some(1500));
}

#[tokio::test]
async fn http_error_emits_llm_call_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let emitter = Arc::new(StatusEmitter::new());
    let collector = Arc::new(CollectingSubscriber::new());
    emitter
        .subscribe(Arc::clone(&collector) as Arc<dyn StatusSubscriber>)
        .await;

    let options = opts(server.uri(), Arc::clone(&emitter));
    let _err = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &options,
    )
    .await
    .expect_err("should fail on 500");

    let events = collector.take_events().await;
    assert!(events
        .iter()
        .any(|e| e.kind == ProcessEventKind::LlmCallError));
}
