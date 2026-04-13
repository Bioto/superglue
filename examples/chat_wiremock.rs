//! Local mock OpenAI: same flow as integration tests, no API key (uses [`wiremock`]).
//!
//! Run: `cargo run --example chat_wiremock --features example-wiremock`

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::hooks::HookRegistry;
use superglue::guardrails::GuardrailRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{Tool, ToolRegistry, ToolSpec};

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

fn text_only() -> serde_json::Value {
    json!({
        "id": "chatcmpl-mock",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok from mock server" },
            "finish_reason": "stop"
        }]
    })
}

fn with_tool_call() -> serde_json::Value {
    json!({
        "id": "chatcmpl-tool",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": "{\"message\":\"hi\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = if i == 0 {
                with_tool_call()
            } else {
                text_only()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default())?;
    let registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool)).await?;

    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-mock".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let messages = vec![ChatMessage::text("user", "trigger tool then finish")];

    let out = complete_with_tools(&http, &registry, &HookRegistry::new(), &GuardrailRegistry::new(), messages, &opts).await?;
    println!("rounds: {}", out.rounds);
    println!("content: {:?}", out.content);
    Ok(())
}
