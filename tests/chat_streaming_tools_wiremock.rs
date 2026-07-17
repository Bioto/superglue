//! Streaming tool-round tests (OpenAI SSE format).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::chat::{ChatOptions, stream_complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
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

fn tool_round_sse() -> String {
    let mut body = String::new();
    let chunks = [
        json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": ""}
                    }]
                },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    ];
    for c in chunks {
        body.push_str(&format!("data: {c}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn text_round_sse(tokens: &[&str]) -> String {
    let mut body = String::new();
    for t in tokens {
        let c = json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {"content": t},
                "finish_reason": null
            }]
        });
        body.push_str(&format!("data: {c}\n\n"));
    }
    let final_c = json!({
        "id": "chatcmpl-s",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });
    body.push_str(&format!("data: {final_c}\n\n"));
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn stream_tool_round_then_text() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let sse = if i == 0 {
                tool_round_sse()
            } else {
                text_round_sse(&["done"])
            };
            ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let mut deltas = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &opts,
        |d| deltas.push(d),
    )
    .await
    .unwrap();

    assert_eq!(out.content, "done");
    assert_eq!(out.rounds, 2);
    assert_eq!(deltas.join(""), "done");
}
