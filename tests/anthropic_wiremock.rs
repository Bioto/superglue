//! Anthropic Messages API completions against wiremock.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::providers::ProviderCredentials;
use superglue::tools::{Tool, ToolRegistry, ToolSpec};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: None,
            parameters_schema: json!({"type": "object"}),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

fn anthropic_text_response() -> serde_json::Value {
    json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello anthropic"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

fn anthropic_tool_response() -> serde_json::Value {
    json!({
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "echo",
            "input": {"x": 1}
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    })
}

fn anthropic_opts(server_uri: &str) -> ChatOptions {
    let mut creds = ProviderCredentials::new();
    creds.insert_key(superglue::providers::ProviderId::Anthropic, "sk-ant-test");
    creds.insert_base_url(superglue::providers::ProviderId::Anthropic, server_uri);
    ChatOptions {
        base_url: server_uri.to_string(),
        api_key: secrecy::Secret::new("legacy".into()),
        model: "anthropic:claude-sonnet-4-20250514".into(),
        max_tool_rounds: 4,
        provider_credentials: Some(std::sync::Arc::new(creds)),
        ..Default::default()
    }
}

#[tokio::test]
async fn anthropic_text_completion() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response()))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let out = complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &anthropic_opts(&server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello anthropic"));
    assert_eq!(out.rounds, 1);
}

#[tokio::test]
async fn anthropic_tool_round_then_text() {
    let server = MockServer::start().await;
    let n = std::sync::atomic::AtomicU32::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let body = if i == 0 {
                anthropic_tool_response()
            } else {
                anthropic_text_response()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "echo")],
        &anthropic_opts(&server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello anthropic"));
    assert_eq!(out.rounds, 2);
}
