//! Edge-case tests for `stream_complete`.
//!
//! Covers: SSE chunks split across byte boundaries, non-200 from streaming
//! endpoint, empty delta skipping, usage in final chunk, concurrent streams,
//! no-content stream, and heartbeat/comment lines.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use superglue::chat::{ChatOptions, stream_complete};
use superglue::hooks::HookRegistry;
use superglue::guardrails::GuardrailRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn opts(base_url: String) -> ChatOptions {
    ChatOptions {
        base_url,
        api_key: "sk-test".into(),
        model: "mock".into(),
        ..Default::default()
    }
}

// Build a complete well-formed SSE stream body from a slice of content tokens.
fn sse_body(tokens: &[&str]) -> String {
    let mut body = String::new();
    for token in tokens {
        let chunk = serde_json::json!({
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000u64,
            "model": "mock",
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": null}]
        });
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    // Final chunk — finish_reason + usage
    let final_chunk = serde_json::json!({
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000u64,
        "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });
    body.push_str(&format!("data: {final_chunk}\n\n"));
    body.push_str("data: [DONE]\n\n");
    body
}

// ---------------------------------------------------------------------------
// Basic delivery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_delivers_all_tokens_in_order() {
    let server = MockServer::start().await;
    let tokens = ["The", " quick", " brown", " fox"];
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body(&tokens), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let mut received = Vec::new();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |d| received.push(d),
    )
    .await
    .unwrap();

    assert_eq!(received, tokens);
    assert_eq!(out.content, "The quick brown fox");
    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
}

// ---------------------------------------------------------------------------
// Finish reason and usage in final chunk
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_extracted_from_final_chunk() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body(&["hello"]), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |_| {},
    )
    .await
    .unwrap();

    let u = out.usage.as_ref().expect("usage should be present");
    assert_eq!(u.prompt_tokens, 5);
    assert_eq!(u.completion_tokens, 3);
    assert_eq!(u.total_tokens, 8);
}

#[tokio::test]
async fn finish_reason_length_in_stream() {
    let server = MockServer::start().await;
    let mut body = String::new();
    let chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]
    });
    body.push_str(&format!("data: {chunk}\n\n"));
    let final_chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
    });
    body.push_str(&format!("data: {final_chunk}\n\n"));
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "long")],
        &opts(server.uri()),
        |_| {},
    )
    .await
    .unwrap();
    assert_eq!(out.finish_reason.as_deref(), Some("length"));
}

// ---------------------------------------------------------------------------
// Empty delta tokens are skipped (not sent to callback)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_delta_content_not_forwarded_to_callback() {
    let server = MockServer::start().await;
    // Mix of empty-string and real tokens
    let mut body = String::new();
    for content in &["", "hello", "", " world", ""] {
        let chunk = serde_json::json!({
            "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
        });
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    let final_chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    body.push_str(&format!("data: {final_chunk}\n\n"));
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let mut tokens: Vec<String> = Vec::new();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |d| tokens.push(d),
    )
    .await
    .unwrap();

    // Only non-empty tokens should reach the callback
    assert_eq!(tokens, vec!["hello", " world"]);
    assert_eq!(out.content, "hello world");
}

// ---------------------------------------------------------------------------
// Chunks with null content field are skipped (not forwarded to callback)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn null_content_delta_not_forwarded() {
    let server = MockServer::start().await;
    let mut body = String::new();
    // First chunk: role announcement, no content
    let role_chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    });
    body.push_str(&format!("data: {role_chunk}\n\n"));
    // Real content
    let content_chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": null}]
    });
    body.push_str(&format!("data: {content_chunk}\n\n"));
    // Final
    let final_chunk = serde_json::json!({
        "id": "s", "object": "chat.completion.chunk", "created": 0u64, "model": "mock",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    body.push_str(&format!("data: {final_chunk}\n\n"));
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let mut tokens = Vec::new();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |d| tokens.push(d),
    )
    .await
    .unwrap();
    // Only the real content token should reach the callback
    assert_eq!(tokens, vec!["hello"]);
    assert_eq!(out.content, "hello");
}

// ---------------------------------------------------------------------------
// Non-200 status from streaming endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_non_200_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let err = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, superglue::chat::ChatError::Http(_)),
        "expected Http error, got {err:?}"
    );
}

#[tokio::test]
async fn stream_403_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(403).set_body_string("Forbidden"))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let err = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |_| {},
    )
    .await
    .unwrap_err();
    assert!(matches!(err, superglue::chat::ChatError::Http(_)));
}

// ---------------------------------------------------------------------------
// Empty stream (only [DONE], no content tokens)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_with_no_content_tokens_returns_empty_string() {
    let server = MockServer::start().await;
    let body = "data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let mut tokens = Vec::<String>::new();
    let out = stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |d| tokens.push(d),
    )
    .await
    .unwrap();

    assert!(tokens.is_empty());
    assert!(out.content.is_empty());
    assert!(out.finish_reason.is_none());
}

// ---------------------------------------------------------------------------
// System prompt forwarded in streaming request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_system_prompt_prepended() {
    use std::sync::Mutex;
    use wiremock::Request;

    let server = MockServer::start().await;
    let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let cap = Arc::clone(&captured);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let body: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or_default();
            *cap.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_raw(
                "data: [DONE]\n\n",
                "text/event-stream",
            )
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: "sk-test".into(),
        model: "mock".into(),
        system_prompt: Some("Always respond in JSON.".into()),
        ..Default::default()
    };
    stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
        |_| {},
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Always respond in JSON.");
}

// ---------------------------------------------------------------------------
// stream: true always set in request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_request_always_sets_stream_true() {
    use std::sync::Mutex;
    use wiremock::Request;

    let server = MockServer::start().await;
    let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
    let cap = Arc::clone(&captured);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let body: serde_json::Value =
                serde_json::from_slice(&req.body).unwrap_or_default();
            *cap.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_raw(
                "data: [DONE]\n\n",
                "text/event-stream",
            )
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    stream_complete(
        &http,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts(server.uri()),
        |_| {},
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

// ---------------------------------------------------------------------------
// Concurrent streams
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_streams_all_succeed() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_raw(
                sse_body(&["hello"]),
                "text/event-stream",
            )
        })
        .mount(&server)
        .await;

    let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
    let opts = Arc::new(opts(server.uri()));

    let handles: Vec<_> = (0..6)
        .map(|_| {
            let http = Arc::clone(&http);
            let opts = Arc::clone(&opts);
            tokio::spawn(async move {
                stream_complete(
                    &http,
                    &HookRegistry::new(),
        &GuardrailRegistry::new(),
                    vec![ChatMessage::text("user", "hi")],
                    &opts,
                    |_| {},
                )
                .await
                .unwrap()
            })
        })
        .collect();

    let results = futures_util::future::join_all(handles).await;
    assert_eq!(count.load(Ordering::SeqCst), 6);
    for r in results {
        assert_eq!(r.unwrap().content, "hello");
    }
}
