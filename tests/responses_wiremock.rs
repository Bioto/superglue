//! Responses API wiremock integration tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::chat::ChatOptions;
use superglue::responses::{complete_with_tools, stream_response};
use superglue::tools::{Tool, ToolRegistry, ToolSpec};

fn http_client() -> HttpClient {
    HttpClient::new(ClientConfig::default()).unwrap()
}

fn opts(base_url: String) -> ChatOptions {
    ChatOptions {
        base_url,
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    }
}

fn text_response(content: &str) -> serde_json::Value {
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

fn tool_call_response() -> serde_json::Value {
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

fn final_after_tool() -> serde_json::Value {
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

#[tokio::test]
async fn responses_text_only() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("hello responses")))
        .mount(&server)
        .await;

    let out = complete_with_tools(
        &http_client(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &opts(server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("hello responses"));
    assert_eq!(out.rounds, 1);
    assert_eq!(out.id, "resp_text");
}

#[tokio::test]
async fn responses_tool_round_threads_previous_response_id() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = serde_json::from_slice::<serde_json::Value>(&req.body)
                .unwrap_or(json!({}));
            if i == 0 {
                assert!(body.get("previous_response_id").is_none());
                ResponseTemplate::new(200).set_body_json(tool_call_response())
            } else {
                assert_eq!(
                    body.get("previous_response_id").and_then(|v| v.as_str()),
                    Some("resp_tool")
                );
                let input = body.get("input").expect("tool round input");
                assert!(input.is_array());
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
        &opts(server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("done"));
    assert_eq!(out.rounds, 2);
}

#[tokio::test]
async fn responses_stream_text_deltas() {
    let server = MockServer::start().await;
    let mut body = String::new();
    body.push_str(
        &format!(
            "data: {}\n\n",
            json!({
                "type": "response.created",
                "response": { "id": "resp_stream" }
            })
        ),
    );
    for token in ["Hi", " there"] {
        body.push_str(
            &format!(
                "data: {}\n\n",
                json!({
                    "type": "response.output_text.delta",
                    "delta": token
                })
            ),
        );
    }
    body.push_str(
        &format!(
            "data: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_stream",
                    "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 }
                }
            })
        ),
    );

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let mut deltas = Vec::new();
    let out = stream_response(
        &http_client(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        "hi",
        &opts(server.uri()),
        |d| deltas.push(d),
    )
    .await
    .unwrap();

    assert_eq!(deltas, ["Hi", " there"]);
    assert_eq!(out.content, "Hi there");
    assert_eq!(out.id, "resp_stream");
}
