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
            static_tool: false,
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
        api_key: secrecy::SecretString::from("legacy"),
        model: "anthropic:claude-sonnet-4-20250514".into(),
        max_tool_rounds: 4,
        provider_credentials: Some(std::sync::Arc::new(creds)),
        system_prompt_blocks: Some(vec![superglue::chat::SystemPromptBlock::cached(
            "stable system",
        )]),
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
async fn anthropic_request_includes_prompt_cache_markers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(|req: &wiremock::Request| {
            let body: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or(json!({}));
            let system = body
                .get("system")
                .and_then(|s| s.as_array())
                .expect("system blocks");
            assert_eq!(
                system[0]["cache_control"]["type"].as_str(),
                Some("ephemeral")
            );
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_cache",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "cached ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 80
                }
            }))
        })
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

    assert_eq!(out.content.as_deref(), Some("cached ok"));
    assert_eq!(out.usage.as_ref().and_then(|u| u.cached_tokens), Some(80));
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
