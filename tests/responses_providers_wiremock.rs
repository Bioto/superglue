//! Multi-provider Responses API wiremock tests (xAI, Groq, Anthropic).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::providers::{ProviderCredentials, ProviderId};
use superglue::responses::{complete_with_tools, stream_response};
use superglue::tools::{Tool, ToolRegistry, ToolSpec};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn http_client() -> HttpClient {
    HttpClient::new(ClientConfig::default()).unwrap()
}

fn provider_opts(server_uri: &str, provider: ProviderId, model: &str) -> ChatOptions {
    let mut creds = ProviderCredentials::new();
    creds.insert_key(provider, "test-key");
    creds.insert_base_url(provider, server_uri);
    ChatOptions {
        base_url: server_uri.to_string(),
        api_key: secrecy::SecretString::from("legacy"),
        model: model.into(),
        max_tool_rounds: 4,
        provider_credentials: Some(Arc::new(creds)),
        ..Default::default()
    }
}

fn text_response(content: &str) -> Value {
    json!({
        "id": "resp_text",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": content }]
        }],
        "usage": { "input_tokens": 3, "output_tokens": 5, "total_tokens": 8 }
    })
}

fn tool_call_response() -> Value {
    json!({
        "id": "resp_tool",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": "{\"x\":1}"
        }]
    })
}

fn final_after_tool() -> Value {
    json!({
        "id": "resp_final",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "done" }]
        }],
        "usage": { "input_tokens": 10, "output_tokens": 2, "total_tokens": 12 }
    })
}

fn anthropic_text_response() -> Value {
    json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello anthropic responses"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

fn anthropic_tool_response() -> Value {
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

fn anthropic_thinking_tool_response() -> Value {
    json!({
        "id": "msg_think_tool",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "need echo",
                "signature": "sig_abc"
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "echo",
                "input": {"x": 1}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    })
}

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

    async fn call(&self, arguments: Value) -> Result<Value, superglue::tools::ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

#[tokio::test]
async fn responses_xai_posts_v1_responses_with_bearer() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(|req: &wiremock::Request| {
            let auth = req
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            assert!(auth.starts_with("Bearer "));
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
            assert_eq!(body.get("model").and_then(|v| v.as_str()), Some("grok-4"));
            ResponseTemplate::new(200).set_body_json(text_response("xai ok"))
        })
        .mount(&server)
        .await;

    let out = complete_with_tools(
        &http_client(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &provider_opts(&server.uri(), ProviderId::Xai, "xai:grok-4"),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("xai ok"));
}

#[tokio::test]
async fn responses_xai_sets_x_grok_conv_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(|req: &wiremock::Request| {
            let conv = req
                .headers
                .get("x-grok-conv-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            assert_eq!(conv, "session-abc");
            ResponseTemplate::new(200).set_body_json(text_response("cached"))
        })
        .mount(&server)
        .await;

    let mut opts = provider_opts(&server.uri(), ProviderId::Xai, "xai:grok-4");
    opts.prompt_cache_key = Some("session-abc".into());

    let out = complete_with_tools(
        &http_client(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &opts,
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("cached"));
}

#[tokio::test]
async fn responses_groq_tool_round_full_history_no_prev_id() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
            if i == 0 {
                assert!(body.get("previous_response_id").is_none());
                ResponseTemplate::new(200).set_body_json(tool_call_response())
            } else {
                assert!(body.get("previous_response_id").is_none());
                assert!(body.get("store").is_none());
                assert!(body.get("prompt_cache_key").is_none());
                let input = body
                    .get("input")
                    .and_then(|v| v.as_array())
                    .expect("input array");
                assert!(
                    input.iter().any(|item| {
                        item.get("type").and_then(|t| t.as_str()) == Some("message")
                    }),
                    "Groq must send full history with message items, got {input:?}"
                );
                assert!(
                    input.iter().any(|item| {
                        item.get("type").and_then(|t| t.as_str()) == Some("function_call")
                    }),
                    "full history should include function_call, got {input:?}"
                );
                ResponseTemplate::new(200).set_body_json(final_after_tool())
            }
        })
        .mount(&server)
        .await;

    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();

    let out = complete_with_tools(
        &http_client(),
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "use echo",
        &provider_opts(
            &server.uri(),
            ProviderId::Groq,
            "groq:llama-3.3-70b-versatile",
        ),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("done"));
    assert_eq!(out.rounds, 2);
}

#[tokio::test]
async fn anthropic_responses_text_posts_messages() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response()))
        .mount(&server)
        .await;

    let out = complete_with_tools(
        &http_client(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &provider_opts(
            &server.uri(),
            ProviderId::Anthropic,
            "anthropic:claude-sonnet-4-20250514",
        ),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello anthropic responses"));
    assert_eq!(out.rounds, 1);
}

#[tokio::test]
async fn anthropic_responses_tool_round_then_text() {
    let server = MockServer::start().await;
    let n = AtomicU32::new(0);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n.fetch_add(1, Ordering::SeqCst);
            let body = if i == 0 {
                anthropic_tool_response()
            } else {
                anthropic_text_response()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();

    let out = complete_with_tools(
        &http_client(),
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "echo",
        &provider_opts(
            &server.uri(),
            ProviderId::Anthropic,
            "anthropic:claude-sonnet-4-20250514",
        ),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello anthropic responses"));
    assert_eq!(out.rounds, 2);
}

#[tokio::test]
async fn anthropic_responses_thinking_budget_in_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(|req: &wiremock::Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
            assert_eq!(body["thinking"]["type"].as_str(), Some("enabled"));
            assert!(body["thinking"]["budget_tokens"].as_u64().unwrap_or(0) > 0);
            assert!(body.get("previous_response_id").is_none());
            ResponseTemplate::new(200).set_body_json(anthropic_text_response())
        })
        .mount(&server)
        .await;

    let mut opts = provider_opts(
        &server.uri(),
        ProviderId::Anthropic,
        "anthropic:claude-sonnet-4-20250514",
    );
    opts.reasoning_effort = Some("medium".into());
    opts.max_completion_tokens = Some(8192);

    complete_with_tools(
        &http_client(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "think",
        &opts,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn anthropic_responses_stream_text() {
    let server = MockServer::start().await;
    let mut body = String::new();
    body.push_str(&format!(
        "data: {}\n\n",
        json!({
            "type": "message_start",
            "message": { "usage": { "input_tokens": 5, "output_tokens": 0 } }
        })
    ));
    body.push_str(&format!(
        "data: {}\n\n",
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": "Hi" }
        })
    ));
    body.push_str(&format!(
        "data: {}\n\n",
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": " there" }
        })
    ));
    body.push_str(&format!("data: {}\n\n", json!({ "type": "message_stop" })));

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let mut deltas = Vec::new();
    let out = stream_response(
        &http_client(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &provider_opts(
            &server.uri(),
            ProviderId::Anthropic,
            "anthropic:claude-sonnet-4-20250514",
        ),
        |d| deltas.push(d),
    )
    .await
    .unwrap();

    assert_eq!(deltas, ["Hi", " there"]);
    assert_eq!(out.content, "Hi there");
}

#[tokio::test]
async fn anthropic_responses_preserves_thinking_blocks_after_tool_round() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(move |req: &wiremock::Request| {
            let i = counter.fetch_add(1, Ordering::SeqCst);
            let body: Value = serde_json::from_slice(&req.body).unwrap_or(json!({}));
            if i > 0 {
                let messages = body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .expect("messages");
                let has_thinking = messages.iter().any(|m| {
                    m.get("content")
                        .and_then(|c| c.as_array())
                        .is_some_and(|blocks| {
                            blocks
                                .iter()
                                .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
                        })
                });
                let has_tool_use = messages.iter().any(|m| {
                    m.get("content")
                        .and_then(|c| c.as_array())
                        .is_some_and(|blocks| {
                            blocks
                                .iter()
                                .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
                        })
                });
                assert!(
                    has_thinking,
                    "second round must preserve thinking blocks: {messages:?}"
                );
                assert!(
                    has_tool_use,
                    "second round must preserve tool_use blocks: {messages:?}"
                );
            }
            let resp = if i == 0 {
                anthropic_thinking_tool_response()
            } else {
                anthropic_text_response()
            };
            ResponseTemplate::new(200).set_body_json(resp)
        })
        .mount(&server)
        .await;

    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();

    let out = complete_with_tools(
        &http_client(),
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "echo with thinking",
        &provider_opts(
            &server.uri(),
            ProviderId::Anthropic,
            "anthropic:claude-sonnet-4-20250514",
        ),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello anthropic responses"));
    assert_eq!(out.rounds, 2);
    assert!(n.load(Ordering::SeqCst) >= 2);
}
