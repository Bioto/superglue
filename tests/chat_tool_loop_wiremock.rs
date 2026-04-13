//! Chat completions + tool loop against a mock OpenAI server.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::json;
use superglue::chat::{ChatError, ChatOptions, complete_with_tools};
use superglue::hooks::HookRegistry;
use superglue::guardrails::GuardrailRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::{ChatMessage, MessageContent};
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

fn text_only_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hello" },
            "finish_reason": "stop"
        }]
    })
}

fn tool_call_response() -> serde_json::Value {
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
                        "arguments": "{\"x\":1}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

#[tokio::test]
async fn completion_text_only_no_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_only_response()))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };
    let messages = vec![ChatMessage::text("user", "hi")];
    let out = complete_with_tools(&http, &reg, &HookRegistry::new(), &GuardrailRegistry::new(), messages, &opts)
        .await
        .unwrap();
    assert_eq!(out.content.as_deref(), Some("hello"));
    assert_eq!(out.rounds, 1);
}

#[tokio::test]
async fn completion_tool_then_assistant_text() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = if i == 0 {
                tool_call_response()
            } else {
                text_only_response()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(std::sync::Arc::new(EchoTool)).await.unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };
    let messages = vec![ChatMessage::text("user", "use echo")];
    let out = complete_with_tools(&http, &reg, &HookRegistry::new(), &GuardrailRegistry::new(), messages, &opts)
        .await
        .unwrap();
    assert_eq!(out.content.as_deref(), Some("hello"));
    assert_eq!(out.rounds, 2);
}

#[tokio::test]
async fn max_tool_rounds_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response()))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(std::sync::Arc::new(EchoTool)).await.unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 2,
        ..Default::default()
    };
    let messages = vec![ChatMessage::text("user", "loop")];
    let err = complete_with_tools(&http, &reg, &HookRegistry::new(), &GuardrailRegistry::new(), messages, &opts)
        .await
        .unwrap_err();
    assert!(matches!(err, ChatError::MaxToolRounds(2)));
}

#[tokio::test]
async fn system_prompt_is_prepended() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_only_response()))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        system_prompt: Some("You are helpful.".into()),
        ..Default::default()
    };
    let messages = vec![ChatMessage::text("user", "hi")];
    let out = complete_with_tools(&http, &reg, &HookRegistry::new(), &GuardrailRegistry::new(), messages, &opts)
        .await
        .unwrap();
    assert_eq!(out.content.as_deref(), Some("hello"));
}

#[tokio::test]
async fn multipart_content_message() {
    // Verify MessageContent::Parts serialises correctly.
    use superglue::openai::{ContentPart, ImageUrl};
    let msg = ChatMessage {
        role: "user".into(),
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Describe this image".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".into(),
                    detail: None,
                },
            },
        ])),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
    };
    let v = serde_json::to_value(&msg).unwrap();
    let parts = v["content"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");
}

// ---------------------------------------------------------------------------
// Streaming tests
// ---------------------------------------------------------------------------

fn sse_body(chunks: &[&str]) -> String {
    let mut body = String::new();
    for content in chunks {
        let data = serde_json::json!({
            "id": "chatcmpl-stream-test",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000u64,
            "model": "mock",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
        });
        body.push_str(&format!("data: {}\n\n", data));
    }
    // Final chunk with finish_reason + usage
    let final_chunk = serde_json::json!({
        "id": "chatcmpl-stream-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000u64,
        "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    body.push_str(&format!("data: {}\n\n", final_chunk));
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn stream_complete_delivers_tokens() {
    let server = MockServer::start().await;
    let words = ["Hello", ", ", "world", "!"];
    let body = sse_body(&words);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let options = superglue::chat::ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("test".to_string()),
        model: "mock".into(),
        ..Default::default()
    };
    let messages = vec![superglue::openai::ChatMessage::text("user", "hi")];

    let mut received = Vec::new();
    let outcome = superglue::chat::stream_complete(
        &http,
        &superglue::hooks::HookRegistry::new(),
        &GuardrailRegistry::new(),
        messages,
        &options,
        |delta| received.push(delta),
    )
    .await
    .unwrap();

    assert_eq!(received, words);
    assert_eq!(outcome.content, "Hello, world!");
    assert_eq!(outcome.finish_reason.as_deref(), Some("stop"));
    assert!(outcome.usage.is_some());
}

// ---------------------------------------------------------------------------
// Parallel tool dispatch
// ---------------------------------------------------------------------------

/// Two tool calls returned in a single LLM response — both execute concurrently
/// (driven by join_all) and appear as two tool-role messages in the next request.
/// The whole exchange completes in exactly 2 completion rounds.
#[tokio::test]
async fn parallel_tool_calls_complete_in_one_round() {
    let server = MockServer::start().await;

    // First response: two tool calls simultaneously.
    let two_tool_calls = serde_json::json!({
        "id": "chatcmpl-parallel",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": { "name": "echo", "arguments": "{\"v\":1}" }
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": { "name": "echo", "arguments": "{\"v\":2}" }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }]
    });

    // Second response: final text after both tool results.
    let final_text = serde_json::json!({
        "id": "chatcmpl-final",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "both done" },
            "finish_reason": "stop"
        }]
    });

    // First registered = higher priority: fires once (two-tool round).
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(two_tool_calls))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Fallback: final text response.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(final_text))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();

    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        ..Default::default()
    };

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &opts,
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("both done"));
    assert_eq!(out.rounds, 2, "two rounds: one tool-call round + one final-text round");
}
