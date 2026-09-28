//! Error-path tests for `complete_with_tools`.
//!
//! Covers: HTTP error statuses, empty choices, malformed JSON in tool
//! arguments, unknown tools, options forwarding, concurrent calls, and
//! usage/finish_reason extraction.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};
use superglue::chat::{ChatError, ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn default_opts(base_url: String) -> ChatOptions {
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

fn tool_call_response(tool: &str, args: &str) -> serde_json::Value {
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
                    "function": {"name": tool, "arguments": args}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn tool_call_response_length_truncated(tool: &str, args: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-tool-trunc",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_trunc",
                    "type": "function",
                    "function": {"name": tool, "arguments": args}
                }]
            },
            "finish_reason": "length"
        }]
    })
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".into(),
            description: None,
            parameters_schema: json!({"type": "object"}),
            static_tool: false,
        }
    }
    async fn call(&self, args: Value) -> Result<Value, ToolInvokeError> {
        Ok(json!({"echo": args}))
    }
}

struct ErrorTool;

#[async_trait]
impl Tool for ErrorTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "error_tool".into(),
            description: None,
            parameters_schema: json!({}),
            static_tool: false,
        }
    }
    async fn call(&self, _: Value) -> Result<Value, ToolInvokeError> {
        Err(ToolInvokeError::handler("boom", Some("ERR".into())))
    }
}

// ---------------------------------------------------------------------------
// HTTP error statuses from the LLM endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_401_returns_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err.root_cause(), ChatError::Http(_)),
        "expected Http error, got {err:?}"
    );
}

#[tokio::test]
async fn http_403_returns_error_not_retried() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(403).set_body_string("Forbidden")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert!(matches!(err.root_cause(), ChatError::Http(_)));
    assert_eq!(count.load(Ordering::SeqCst), 1, "403 must not be retried");
}

#[tokio::test]
async fn http_500_returns_error_not_retried() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(500).set_body_string("Server error")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert_eq!(count.load(Ordering::SeqCst), 1, "500 must not be retried");
}

#[tokio::test]
async fn http_429_is_retried_and_eventually_succeeds() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            let n = c.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                ResponseTemplate::new(429).set_body_string("Rate limited")
            } else {
                ResponseTemplate::new(200).set_body_json(text_response("ok"))
            }
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap();
    assert_eq!(out.content.as_deref(), Some("ok"));
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// Empty choices
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_choices_returns_no_choice_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-empty",
            "model": "mock",
            "choices": []
        })))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err.root_cause(), ChatError::NoChoice),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Malformed JSON in tool arguments
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_tool_args_returns_error_to_model_not_turn_failure() {
    // When the LLM emits malformed tool-call JSON the error is returned as a
    // soft tool-result message so the model can retry, rather than killing the
    // turn immediately with ChatError::Serde.
    //
    // Since the mock always returns the same bad call the loop exhausts its
    // rounds, but the turn should NOT fail with Serde — it fails with MaxToolRounds
    // (or a text reply if the model recovers), never with a Serde parse error.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(tool_call_response("echo", "NOT VALID JSON {{{")),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let result = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "call echo")],
        &default_opts(server.uri()),
    )
    .await;
    assert!(
        !matches!(
            result.as_ref().err().map(|e| e.root_cause()),
            Some(ChatError::Serde(_))
        ),
        "malformed tool args should not propagate as ChatError::Serde; got {result:?}"
    );
    // The mock keeps returning bad args so the run exhausts rounds.
    assert!(
        matches!(
            result.as_ref().err().map(|e| e.root_cause()),
            Some(ChatError::MaxToolRounds(_))
        ) || result.is_ok(),
        "expected MaxToolRounds or success (model recovers), got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Unknown tool name
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_tool_returns_tool_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(tool_call_response("nonexistent_tool", "{}")),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    // Registry is empty — no tools registered
    let reg = ToolRegistry::new();
    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "call missing")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            err.root_cause(),
            ChatError::Tool(ToolInvokeError::UnknownTool { .. })
        ),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Tool handler error propagates
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_handler_error_propagates() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(tool_call_response("error_tool", "{}")),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(ErrorTool)).await.unwrap();
    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "trigger error")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            err.root_cause(),
            ChatError::Tool(ToolInvokeError::HandlerFailed { .. })
        ),
        "got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Options forwarded into request body
// ---------------------------------------------------------------------------

#[tokio::test]
async fn temperature_forwarded_to_request() {
    use std::sync::Mutex;
    use wiremock::Request;

    let server = MockServer::start().await;
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let cap = Arc::clone(&captured);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *cap.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_json(text_response("hi"))
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        temperature: Some(0.42),
        max_completion_tokens: Some(100),
        seed: Some(99),
        ..Default::default()
    };
    complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &opts,
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let temp = body["temperature"].as_f64().unwrap();
    assert!((temp - 0.42).abs() < 1e-3, "temperature={temp}");
    assert_eq!(body["max_completion_tokens"], 100);
    assert_eq!(body["seed"], 99);
}

#[tokio::test]
async fn no_tools_means_tools_field_absent() {
    use std::sync::Mutex;
    use wiremock::Request;

    let server = MockServer::start().await;
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let cap = Arc::clone(&captured);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *cap.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_json(text_response("hi"))
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new(); // empty
    complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hi")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    assert!(
        body.get("tools").is_none(),
        "tools should be absent when registry is empty"
    );
}

#[tokio::test]
async fn system_prompt_prepended_as_first_message() {
    use std::sync::Mutex;
    use wiremock::Request;

    let server = MockServer::start().await;
    let captured: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let cap = Arc::clone(&captured);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &Request| {
            let body: Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *cap.lock().unwrap() = Some(body);
            ResponseTemplate::new(200).set_body_json(text_response("hi"))
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        system_prompt: Some("Be concise.".into()),
        ..Default::default()
    };
    complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "hello")],
        &opts,
    )
    .await
    .unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert!(messages.len() >= 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Be concise.");
    assert_eq!(messages[1]["role"], "user");
}

// ---------------------------------------------------------------------------
// Usage and finish_reason extraction
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_and_finish_reason_extracted() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-usage",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "ping")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.finish_reason.as_deref(), Some("stop"));
    let u = out.usage.as_ref().unwrap();
    assert_eq!(u.prompt_tokens, 12);
    assert_eq!(u.completion_tokens, 3);
    assert_eq!(u.total_tokens, 15);
}

#[tokio::test]
async fn finish_reason_length_surfaced() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-len",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "truncated…"},
                "finish_reason": "length"
            }]
        })))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "long")],
        &default_opts(server.uri()),
    )
    .await
    .unwrap();
    assert_eq!(out.finish_reason.as_deref(), Some("length"));
}

// When a tool call is truncated (finish_reason=="length"), the streaming loop
// must NOT try to parse the incomplete JSON. Instead it returns a soft error
// to the model, which then recovers with a text reply.
#[tokio::test]
async fn finish_reason_length_with_tool_call_returns_truncation_error_to_model() {
    let server = MockServer::start().await;

    // First call: truncated tool-call (finish_reason=length, incomplete args)
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            tool_call_response_length_truncated("echo", r#"{"input": "he"#),
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second call: model recovers with a text reply
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("I'll retry")))
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
        vec![ChatMessage::text("user", "echo something")],
        &default_opts(server.uri()),
    )
    .await
    .expect("should not fail — truncation must be soft-errored to model, not kill the turn");

    assert_eq!(out.content.as_deref(), Some("I'll retry"));
}

// ---------------------------------------------------------------------------
// Concurrent complete_with_tools calls share the same HttpClient
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_completions_all_succeed() {
    let server = MockServer::start().await;
    let count = Arc::new(AtomicU32::new(0));
    let c = Arc::clone(&count);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_: &wiremock::Request| {
            c.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(text_response("hi"))
        })
        .mount(&server)
        .await;

    let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
    let reg = Arc::new(ToolRegistry::new());
    let opts = Arc::new(default_opts(server.uri()));

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let http = Arc::clone(&http);
            let reg = Arc::clone(&reg);
            let opts = Arc::clone(&opts);
            tokio::spawn(async move {
                complete_with_tools(
                    &http,
                    &reg,
                    &HookRegistry::new(),
                    &GuardrailRegistry::new(),
                    vec![ChatMessage::text("user", "hi")],
                    &opts,
                )
                .await
                .unwrap()
            })
        })
        .collect();

    let results = futures_util::future::join_all(handles).await;
    assert_eq!(results.len(), 8);
    for r in results {
        assert_eq!(r.unwrap().content.as_deref(), Some("hi"));
    }
    assert_eq!(count.load(Ordering::SeqCst), 8);
}
