//! Responses API streaming tool + reasoning round tests.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::events::{CollectingSubscriber, ProcessEventKind, StatusEmitter, StatusSubscriber};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::responses::stream_complete_with_tools;
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

fn tool_round_sse() -> String {
    let mut body = String::new();
    let events = [
        json!({"type":"response.created","response":{"id":"resp_tool_1"}}),
        json!({"type":"response.reasoning_summary_text.delta","delta":"scan "}),
        json!({"type":"response.reasoning_summary_text.delta","delta":"workspace"}),
        json!({"type":"response.function_call_arguments.done","call_id":"call_1","name":"echo","arguments":"{\"x\":1}"}),
        json!({
            "type":"response.completed",
            "response":{
                "id":"resp_tool_1",
                "usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7},
                "output":[{"type":"function_call","call_id":"call_1","name":"echo","arguments":"{\"x\":1}"}]
            }
        }),
    ];
    for ev in events {
        body.push_str(&format!("data: {ev}\n\n"));
    }
    body
}

fn text_round_sse() -> String {
    let mut body = String::new();
    let events = [
        json!({"type":"response.created","response":{"id":"resp_text_1"}}),
        json!({"type":"response.output_text.delta","delta":"done"}),
        json!({
            "type":"response.completed",
            "response":{
                "id":"resp_text_1",
                "usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4},
                "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]
            }
        }),
    ];
    for ev in events {
        body.push_str(&format!("data: {ev}\n\n"));
    }
    body
}

fn output_item_done_text_round_sse() -> String {
    let mut body = String::new();
    let events = [
        json!({"type":"response.created","response":{"id":"resp_item_1"}}),
        json!({
            "type":"response.output_item.done",
            "output_index":0,
            "item":{
                "type":"message",
                "role":"assistant",
                "status":"completed",
                "content":[{"type":"output_text","text":"hello from item done"}]
            }
        }),
        json!({
            "type":"response.completed",
            "response":{
                "id":"resp_item_1",
                "usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}
            }
        }),
    ];
    for ev in events {
        body.push_str(&format!("data: {ev}\n\n"));
    }
    body
}

fn content_part_done_text_round_sse() -> String {
    let mut body = String::new();
    let chunks = [
        (
            "response.created",
            r#"{"response":{"id":"resp_part_1"}}"#,
        ),
        (
            "response.content_part.done",
            r#"{"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello from content part","annotations":[]}}"#,
        ),
        (
            "response.completed",
            r#"{"response":{"id":"resp_part_1","usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}}}"#,
        ),
    ];
    for (event, data) in chunks {
        body.push_str(&format!("event: {event}\ndata: {data}\n\n"));
    }
    body
}

#[tokio::test]
async fn responses_stream_tool_then_text() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let sse = if i == 0 {
                tool_round_sse()
            } else {
                text_round_sse()
            };
            ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();

    let emitter = Arc::new(StatusEmitter::new());
    let collector = Arc::new(CollectingSubscriber::new());
    emitter
        .subscribe(Arc::clone(&collector) as Arc<dyn StatusSubscriber>)
        .await;

    let opts = superglue::chat::ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        reasoning_effort: Some("medium".into()),
        status_emitter: Some(Arc::clone(&emitter)),
        ..Default::default()
    };

    let mut deltas = Vec::new();
    let mut reasoning = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &opts,
        |d| deltas.push(d),
        |r| reasoning.push(r),
    )
    .await
    .unwrap();

    assert_eq!(out.content, "done");
    assert_eq!(out.rounds, 2);
    assert_eq!(deltas.join(""), "done");
    assert_eq!(reasoning.join(""), "scan workspace");

    let events = collector.take_events().await;
    assert!(events.iter().any(|e| e.kind == ProcessEventKind::ToolCallStart));
    assert!(events.iter().any(|e| e.kind == ProcessEventKind::ToolCallEnd));
    assert!(events.iter().any(|e| e.kind == ProcessEventKind::ReasoningDelta));
}

#[tokio::test]
async fn responses_stream_output_item_done_only_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_req: &wiremock::Request| {
            ResponseTemplate::new(200).set_body_raw(output_item_done_text_round_sse(), "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = superglue::chat::ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let mut deltas = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
        |d| deltas.push(d),
        |_r| {},
    )
    .await
    .unwrap();

    assert_eq!(out.content, "hello from item done");
    assert_eq!(out.rounds, 1);
    assert_eq!(deltas.join(""), "hello from item done");
}

#[tokio::test]
async fn responses_stream_content_part_done_only_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_req: &wiremock::Request| {
            ResponseTemplate::new(200)
                .set_body_raw(content_part_done_text_round_sse(), "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = superglue::chat::ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let mut deltas = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
        |d| deltas.push(d),
        |_r| {},
    )
    .await
    .unwrap();

    assert_eq!(out.content, "hello from content part");
    assert_eq!(out.rounds, 1);
    assert_eq!(deltas.join(""), "hello from content part");
}

fn response_failed_sse() -> String {
    let mut body = String::new();
    let events = [
        json!({"type":"response.created","response":{"id":"resp_fail_1"}}),
        json!({
            "type":"error",
            "error":{
                "type":"invalid_request_error",
                "message":"Your input exceeds the context window"
            }
        }),
        json!({
            "type":"response.failed",
            "response":{
                "id":"resp_fail_1",
                "status":"failed",
                "error":{
                    "code":"invalid_request_error",
                    "message":"Your input exceeds the context window"
                }
            }
        }),
    ];
    for ev in events {
        body.push_str(&format!("data: {ev}\n\n"));
    }
    body
}

#[tokio::test]
async fn responses_stream_response_failed_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_req: &wiremock::Request| {
            ResponseTemplate::new(200).set_body_raw(response_failed_sse(), "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = superglue::chat::ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let err = stream_complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
        |_d| {},
        |_r| {},
    )
    .await
    .unwrap_err();

    let msg = err.to_string();
    assert!(
        msg.contains("invalid_request_error"),
        "expected error code in message, got: {msg}"
    );
    assert!(
        msg.contains("context window"),
        "expected error message in output, got: {msg}"
    );
}
