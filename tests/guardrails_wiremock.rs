//! Guardrails integration tests — all scenarios against a mock OpenAI server.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::batch::{BatchConfig, BatchRequest, ErrorStrategy, batch_complete};
use superglue::chat::{ChatError, ChatOptions, complete_with_tools, stream_complete};
use superglue::guardrails::{
    BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailError, GuardrailHandler,
    GuardrailOutcome, GuardrailRegistry, GuardrailStage, LengthStrategy, MaxLengthGuardrail,
};
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn text_response(text: &str) -> serde_json::Value {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "mock",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    })
}

fn opts(base_url: impl Into<String>) -> ChatOptions {
    ChatOptions::new(base_url, "sk-test", "mock")
}

fn http() -> Arc<HttpClient> {
    Arc::new(HttpClient::new(ClientConfig::default()).unwrap())
}

// ---------------------------------------------------------------------------
// 1. input_guardrail_blocks_before_llm_call
// ---------------------------------------------------------------------------

/// An input guardrail that blocks content containing "badword".
/// The LLM server should never receive a request.
#[tokio::test]
async fn input_guardrail_blocks_before_llm_call() {
    let server = MockServer::start().await;
    // Mount NO mock — if the HTTP call happens, the test will see an unexpected request.
    // We verify by asserting Err(ChatError::Guardrail).

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_input(GuardrailConfig {
            name: "blocklist".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&["badword"], BlocklistAction::Block)
                    .unwrap()
                    .for_stages([GuardrailStage::Input]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "this message has badword in it")];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err.root_cause(),
            ChatError::Guardrail(GuardrailError {
                stage: GuardrailStage::Input,
                ..
            })
        ),
        "expected Guardrail(Input) error, got: {err:?}"
    );

    // Verify zero HTTP calls were made.
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// 2. input_guardrail_transforms_content
// ---------------------------------------------------------------------------

/// A redacting input guardrail modifies the user message; the LLM then replies.
#[tokio::test]
async fn input_guardrail_transforms_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("OK")))
        .mount(&server)
        .await;

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_input(GuardrailConfig {
            name: "redact-secret".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&[r"SECRET-\w+"], BlocklistAction::Redact)
                    .unwrap()
                    .for_stages([GuardrailStage::Input]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text(
        "user",
        "my token is SECRET-abc123 please help",
    )];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    assert!(result.is_ok(), "expected Ok, got: {result:?}");
    // The LLM was called (one HTTP request).
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// 3. output_guardrail_blocks_after_response
// ---------------------------------------------------------------------------

/// Output guardrail rejects the LLM response on the first attempt (retries = 0).
#[tokio::test]
async fn output_guardrail_blocks_after_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(text_response("this has badword in it")),
        )
        .mount(&server)
        .await;

    let guardrails = GuardrailRegistry::new().with_max_output_retries(0); // no retries
    guardrails
        .add_output(GuardrailConfig {
            name: "no-badword".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&["badword"], BlocklistAction::Block)
                    .unwrap()
                    .for_stages([GuardrailStage::Output]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "say something")];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(
            err.root_cause(),
            ChatError::Guardrail(GuardrailError {
                stage: GuardrailStage::Output,
                ..
            })
        ),
        "expected Guardrail(Output) error, got: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. output_guardrail_retries_and_succeeds
// ---------------------------------------------------------------------------

/// First LLM response fails the output guardrail; the second (retry) succeeds.
#[tokio::test]
async fn output_guardrail_retries_and_succeeds() {
    let server = MockServer::start().await;

    // First call: bad response
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("badword is here")))
        .expect(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Retry call: clean response
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("clean response")))
        .mount(&server)
        .await;

    let guardrails = GuardrailRegistry::new().with_max_output_retries(2);
    guardrails
        .add_output(GuardrailConfig {
            name: "no-badword".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&["badword"], BlocklistAction::Block)
                    .unwrap()
                    .for_stages([GuardrailStage::Output]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "say something")];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    let outcome = result.expect("should succeed after retry");
    assert_eq!(outcome.content.as_deref(), Some("clean response"));
}

// ---------------------------------------------------------------------------
// 5. output_guardrail_transforms_content
// ---------------------------------------------------------------------------

/// Output guardrail redacts matching text and returns the transformed content.
#[tokio::test]
async fn output_guardrail_transforms_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(text_response("call me at 555-123-4567")),
        )
        .mount(&server)
        .await;

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_output(GuardrailConfig {
            name: "redact-phone".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&[r"\b\d{3}-\d{3}-\d{4}\b"], BlocklistAction::Redact)
                    .unwrap()
                    .for_stages([GuardrailStage::Output]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "what is your phone?")];
    let outcome = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.content.as_deref(), Some("call me at [REDACTED]"));
}

// ---------------------------------------------------------------------------
// 6. max_length_guardrail_blocks_long_input
// ---------------------------------------------------------------------------

/// MaxLengthGuardrail blocks user messages that are too long.
#[tokio::test]
async fn max_length_guardrail_blocks_long_input() {
    let server = MockServer::start().await;
    // No mock needed — LLM should never be called.

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_input(GuardrailConfig {
            name: "max-len".into(),
            handler: Arc::new(MaxLengthGuardrail::new(
                Some(10),
                None,
                LengthStrategy::Block,
            )),
        })
        .await;

    let messages = vec![ChatMessage::text(
        "user",
        "this message is definitely more than 10 chars",
    )];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.root_cause(),
        ChatError::Guardrail(GuardrailError {
            stage: GuardrailStage::Input,
            ..
        })
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
// ---------------------------------------------------------------------------
// 7. max_length_guardrail_truncates_output
// ---------------------------------------------------------------------------

/// MaxLengthGuardrail with Truncate strategy shortens long output.
#[tokio::test]
async fn max_length_guardrail_truncates_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(text_response("this is a very long response")),
        )
        .mount(&server)
        .await;

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_output(GuardrailConfig {
            name: "max-output".into(),
            handler: Arc::new(MaxLengthGuardrail::new(
                None,
                Some(10),
                LengthStrategy::Truncate,
            )),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "tell me something")];
    let outcome = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await
    .unwrap();

    let content = outcome.content.unwrap();
    assert!(
        content.len() <= 10,
        "content should be truncated to 10 chars, got: {content:?}"
    );
}

// ---------------------------------------------------------------------------
// 8. stream_input_guardrail_blocks
// ---------------------------------------------------------------------------

/// Input guardrail blocks a streaming request before the HTTP call is made.
#[tokio::test]
async fn stream_input_guardrail_blocks() {
    let server = MockServer::start().await;
    // No mock — stream should never start.

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_input(GuardrailConfig {
            name: "blocklist".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&["forbidden"], BlocklistAction::Block)
                    .unwrap()
                    .for_stages([GuardrailStage::Input]),
            ),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "say the forbidden word please")];
    let result = stream_complete(
        &http(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
        |_| {},
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.root_cause(),
        ChatError::Guardrail(GuardrailError {
            stage: GuardrailStage::Input,
            ..
        })
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// 9. batch_input_guardrail_blocks_item
// ---------------------------------------------------------------------------

/// When one item in a batch is blocked by an input guardrail, it becomes a
/// failed BatchResult while other items complete normally (Continue strategy).
#[tokio::test]
async fn batch_input_guardrail_blocks_item() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("ok")))
        .mount(&server)
        .await;

    let guardrails = Arc::new(GuardrailRegistry::new());
    guardrails
        .add_input(GuardrailConfig {
            name: "blocklist".into(),
            handler: Arc::new(
                BlocklistGuardrail::new(&["blocked"], BlocklistAction::Block)
                    .unwrap()
                    .for_stages([GuardrailStage::Input]),
            ),
        })
        .await;

    let requests = vec![
        BatchRequest::new("this is fine"),
        BatchRequest::new("this is blocked content"),
        BatchRequest::new("this is also fine"),
    ];

    let config = BatchConfig {
        max_concurrent: 3,
        error_strategy: ErrorStrategy::Continue,
        ..Default::default()
    };
    let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
    let registry = Arc::new(ToolRegistry::new());
    let hooks = Arc::new(HookRegistry::new());

    let resp = batch_complete(
        http,
        registry,
        hooks,
        guardrails,
        requests,
        &opts(server.uri()),
        config,
    )
    .await
    .unwrap();

    assert_eq!(resp.total_requests, 3);
    assert_eq!(resp.successful, 2, "two items should succeed");
    assert_eq!(resp.failed, 1, "one item should be blocked");

    let blocked = resp.results.iter().find(|r| r.error.is_some());
    assert!(blocked.is_some(), "one result should have an error");
    let err_msg = blocked.unwrap().error.as_deref().unwrap();
    assert!(
        err_msg.contains("guardrail") || err_msg.contains("blocked"),
        "error should mention guardrail: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// 10. custom_handler_blocks
// ---------------------------------------------------------------------------

/// A custom guardrail handler (using a closure via a helper struct) can block content.
struct LengthChecker(usize);

#[async_trait]
impl GuardrailHandler for LengthChecker {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        if content.len() > self.0 {
            GuardrailOutcome::Block(format!("content too long: {} chars", content.len()))
        } else {
            GuardrailOutcome::Allow(content.to_string())
        }
    }
}

#[tokio::test]
async fn custom_handler_blocks() {
    let server = MockServer::start().await;

    let guardrails = GuardrailRegistry::new();
    guardrails
        .add_input(GuardrailConfig {
            name: "custom-checker".into(),
            handler: Arc::new(LengthChecker(5)),
        })
        .await;

    let messages = vec![ChatMessage::text("user", "a long user message")];
    let result = complete_with_tools(
        &http(),
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &guardrails,
        messages,
        &opts(server.uri()),
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(
        err.root_cause(),
        ChatError::Guardrail(GuardrailError {
            stage: GuardrailStage::Input,
            ..
        })
    ));
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
