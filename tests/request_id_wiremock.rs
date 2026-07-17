//! Integration tests for the correlation ID / request_id feature.
//!
//! Four tests:
//!  1. `auto_generated_request_id` — a UUID is returned when none is supplied
//!  2. `caller_supplied_request_id_is_preserved` — custom ID is echoed on outcome
//!  3. `stream_auto_generated_request_id` — streaming variant auto-generates ID
//!  4. `batch_uses_item_id_as_request_id` — batch items use their `.id` as correlation ID

use std::sync::Arc;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::batch::{BatchConfig, BatchRequest, batch_complete};
use superglue::chat::{ChatOptions, complete_with_tools, stream_complete};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn http() -> HttpClient {
    let cfg = ClientConfig {
        retry: superglue::http::RetryPolicy {
            max_retries: 0,
            ..Default::default()
        },
        ..ClientConfig::default()
    };
    HttpClient::new(cfg).unwrap()
}

fn opts(base_url: String) -> ChatOptions {
    ChatOptions {
        base_url,
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        ..Default::default()
    }
}

fn text_response(content: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
}

fn sse_response(content: &str) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({
            "id": "chatcmpl-sse",
            "object": "chat.completion.chunk",
            "model": "mock",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]
        })
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// When no `request_id` is supplied, the outcome contains a non-empty UUID.
#[tokio::test]
async fn auto_generated_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("hello")))
        .mount(&server)
        .await;

    let http = http();
    let opts = opts(server.uri());
    let messages = vec![ChatMessage::text("user", "hi")];
    let registry = ToolRegistry::new();
    let hooks = HookRegistry::new();
    let guardrails = GuardrailRegistry::default();

    let outcome = complete_with_tools(&http, &registry, &hooks, &guardrails, messages, &opts)
        .await
        .expect("complete_with_tools failed");

    // Must be a non-empty string that looks like a UUID (contains hyphens).
    assert!(
        !outcome.request_id.is_empty(),
        "request_id should not be empty"
    );
    assert!(
        outcome.request_id.contains('-'),
        "auto-generated request_id should be a UUID: {}",
        outcome.request_id
    );
}

/// When a `request_id` is supplied via `ChatOptions`, it is echoed on the outcome.
#[tokio::test]
async fn caller_supplied_request_id_is_preserved() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("world")))
        .mount(&server)
        .await;

    let http = http();
    let mut opts = opts(server.uri());
    opts.request_id = Some("my-custom-trace-id".to_string());

    let messages = vec![ChatMessage::text("user", "hi")];
    let registry = ToolRegistry::new();
    let hooks = HookRegistry::new();
    let guardrails = GuardrailRegistry::default();

    let outcome = complete_with_tools(&http, &registry, &hooks, &guardrails, messages, &opts)
        .await
        .expect("complete_with_tools failed");

    assert_eq!(
        outcome.request_id, "my-custom-trace-id",
        "caller-supplied request_id must be echoed on the outcome"
    );
}

/// `stream_complete` also auto-generates a UUID when none is supplied.
#[tokio::test]
async fn stream_auto_generated_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_response("streamed"), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = http();
    let opts = opts(server.uri());
    let messages = vec![ChatMessage::text("user", "hi")];
    let hooks = HookRegistry::new();
    let guardrails = GuardrailRegistry::default();

    let outcome = stream_complete(&http, &hooks, &guardrails, messages, &opts, |_| {})
        .await
        .expect("stream_complete failed");

    assert!(
        !outcome.request_id.is_empty(),
        "stream request_id should not be empty"
    );
    assert!(
        outcome.request_id.contains('-'),
        "auto-generated stream request_id should be a UUID: {}",
        outcome.request_id
    );
}

/// Batch items wire their `id` as the `request_id` in the underlying `complete_with_tools` call.
/// We verify this indirectly: the `BatchResult.id` matches the input `BatchRequest.id`.
#[tokio::test]
async fn batch_uses_item_id_as_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .mount(&server)
        .await;

    let http = Arc::new(http());
    let registry = Arc::new(ToolRegistry::new());
    let hooks = Arc::new(HookRegistry::new());
    let guardrails = Arc::new(GuardrailRegistry::default());
    let opts = opts(server.uri());

    let requests = vec![
        BatchRequest {
            id: Some("req-alpha".to_string()),
            prompt: "hello".to_string(),
            system_prompt: None,
            metadata: std::collections::HashMap::new(),
        },
        BatchRequest {
            id: Some("req-beta".to_string()),
            prompt: "world".to_string(),
            system_prompt: None,
            metadata: std::collections::HashMap::new(),
        },
    ];

    let resp = batch_complete(
        http,
        registry,
        hooks,
        guardrails,
        requests,
        &opts,
        BatchConfig::default(),
    )
    .await
    .expect("batch_complete failed");

    assert_eq!(resp.results.len(), 2);

    let ids: std::collections::HashSet<String> =
        resp.results.iter().map(|r| r.id.clone()).collect();
    assert!(ids.contains("req-alpha"), "req-alpha missing from results");
    assert!(ids.contains("req-beta"), "req-beta missing from results");

    for result in &resp.results {
        assert!(result.success, "all results should succeed");
    }
}
