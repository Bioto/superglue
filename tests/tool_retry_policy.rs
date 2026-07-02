//! Integration tests for per-tool error policies (FailFast / Skip / Retry).
//!
//! These tests mount a wiremock server so the full `complete_with_tools` path
//! runs, exercising the policy dispatch inside the tool loop.

use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use async_trait::async_trait;
use serde_json::{Value, json};
use superglue::chat::{ChatError, ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{
    OnToolError, Tool, ToolInvokeError, ToolRegistry, ToolRetryPolicy, ToolSpec,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helper tools
// ---------------------------------------------------------------------------

/// A tool that always fails.
struct AlwaysFailTool;

#[async_trait]
impl Tool for AlwaysFailTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "always_fail".into(),
            description: None,
            parameters_schema: json!({}),
                    static_tool: false,
        }
    }
    async fn call(&self, _: Value) -> Result<Value, ToolInvokeError> {
        Err(ToolInvokeError::handler("boom", None))
    }
}

/// A tool that fails the first `fail_count` calls then succeeds.
struct FlakeyTool {
    name: String,
    calls: Arc<AtomicU32>,
    fail_count: u32,
}

#[async_trait]
impl Tool for FlakeyTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: None,
            parameters_schema: json!({}),
                    static_tool: false,
        }
    }
    async fn call(&self, _: Value) -> Result<Value, ToolInvokeError> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
        if attempt < self.fail_count {
            Err(ToolInvokeError::handler(
                format!("transient failure #{attempt}"),
                None,
            ))
        } else {
            Ok(json!({"ok": true}))
        }
    }
}

// ---------------------------------------------------------------------------
// Shared response builders
// ---------------------------------------------------------------------------

fn tool_call_response(tool_name: &str) -> Value {
    json!({
        "id": "chatcmpl-tc",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": tool_name, "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn text_response(text: &str) -> Value {
    json!({
        "id": "chatcmpl-txt",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    })
}

fn chat_options(base_url: &str) -> ChatOptions {
    ChatOptions {
        base_url: base_url.to_string(),
        api_key: secrecy::Secret::new("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Default FailFast policy: tool error propagates immediately as ChatError::Tool.
#[tokio::test]
async fn fail_fast_policy_propagates_tool_error() {
    let server = MockServer::start().await;
    // First call returns a tool invocation.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response("always_fail")))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    // Default policy = FailFast.
    reg.register(Arc::new(AlwaysFailTool)).await.unwrap();

    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &chat_options(&server.uri()),
    )
    .await
    .unwrap_err();

    assert!(
        matches!(err, ChatError::Tool(_)),
        "expected ChatError::Tool, got: {err:?}"
    );
}

/// Skip policy: tool error is appended as a tool message; model sees the error
/// and returns a text response — the full outcome is returned without Err.
#[tokio::test]
async fn skip_policy_continues_after_tool_error() {
    let server = MockServer::start().await;
    // First registered = higher priority: returns the tool call once.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response("always_fail")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Second registered = fallback: returns the final text for all subsequent calls.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("recovered")))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register_with_policy(
        Arc::new(AlwaysFailTool),
        ToolRetryPolicy {
            on_error: OnToolError::Skip,
        },
    )
    .await
    .unwrap();

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &chat_options(&server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("recovered"));
    assert_eq!(out.rounds, 2, "one round for tool call, one for final text");
}

/// Retry policy: tool fails once then succeeds on the first retry.
/// The LLM only needs two calls total (one tool-call round, one final-text round).
#[tokio::test]
async fn retry_policy_succeeds_on_second_attempt() {
    let server = MockServer::start().await;
    // First registered = higher priority: returns the tool call once.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response("flakey")))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Second registered = fallback: returns final text.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
        .mount(&server)
        .await;

    let calls = Arc::new(AtomicU32::new(0));
    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register_with_policy(
        Arc::new(FlakeyTool {
            name: "flakey".into(),
            calls: Arc::clone(&calls),
            fail_count: 1,
        }),
        ToolRetryPolicy {
            on_error: OnToolError::Retry {
                max: 3,
                initial_delay_ms: 0,
            },
        },
    )
    .await
    .unwrap();

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &chat_options(&server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("done"));
    // The flakey tool should have been called twice (1 fail + 1 success).
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "expected 2 tool invocations"
    );
}

/// Retry policy exhausts all retries and then propagates the error (FailFast fallback).
#[tokio::test]
async fn retry_policy_exhausted_propagates_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tool_call_response("always_fail")))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register_with_policy(
        Arc::new(AlwaysFailTool),
        ToolRetryPolicy {
            on_error: OnToolError::Retry {
                max: 2,
                initial_delay_ms: 0,
            },
        },
    )
    .await
    .unwrap();

    let err = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &chat_options(&server.uri()),
    )
    .await
    .unwrap_err();

    assert!(
        matches!(err, ChatError::Tool(_)),
        "expected ChatError::Tool after retries exhausted, got: {err:?}"
    );
}

/// `register_with_policy` still rejects duplicate names.
#[tokio::test]
async fn register_with_policy_rejects_duplicates() {
    let reg = ToolRegistry::new();
    reg.register_with_policy(Arc::new(AlwaysFailTool), ToolRetryPolicy::default())
        .await
        .unwrap();

    let err = reg
        .register_with_policy(Arc::new(AlwaysFailTool), ToolRetryPolicy::default())
        .await
        .unwrap_err();

    assert!(
        matches!(
            err,
            superglue::tools::ToolInvokeError::DuplicateRegistration { .. }
        ),
        "expected DuplicateRegistration"
    );
}

/// `policy_for` returns the default for unknown tools.
#[tokio::test]
async fn policy_for_unknown_returns_default() {
    let reg = ToolRegistry::new();
    let policy = reg.policy_for("nonexistent").await;
    assert!(matches!(policy.on_error, OnToolError::FailFast));
}
