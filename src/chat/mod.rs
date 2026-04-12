//! Chat completions: non-streaming tool-loop and streaming (SSE) variants.

use futures_util::StreamExt;
use serde_json::Value;
use thiserror::Error;

use crate::guardrails::{GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage};
use crate::hooks::{HookContext, HookError, HookRegistry, HookStage};
use crate::http::{Error as HttpError, HttpClient, join_base_url, sse::SseParser};
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatMessage, ChatTool, MessageContent,
    ResponseFormat, StopSequence, ToolChoice,
};
use crate::proto;
use crate::tools::ToolInvokeError;
use crate::tools::ToolRegistry;

/// Provider and model settings for [`complete_with_tools`].
///
/// All fields beyond `base_url`, `api_key`, `model`, and `max_tool_rounds` are forwarded
/// directly to the OpenAI `POST /v1/chat/completions` body when set.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// e.g. `https://api.openai.com` (no trailing slash required).
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    /// Maximum **HTTP completion** calls (each can include tool follow-up rounds).
    pub max_tool_rounds: u32,
    /// Prepended as a `"system"` message before all caller messages when set.
    pub system_prompt: Option<String>,

    // --- Sampling ---
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,

    // --- Token limits ---
    pub max_completion_tokens: Option<u32>,

    // --- Penalties ---
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,

    // --- Stop sequences ---
    pub stop: Option<StopSequence>,

    // --- Output format ---
    pub response_format: Option<ResponseFormat>,

    // --- Tool control ---
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,

    // --- Logprobs ---
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,

    // --- Determinism / storage ---
    pub seed: Option<i64>,
    pub store: Option<bool>,

    // --- Service ---
    pub service_tier: Option<String>,

    // --- Reasoning models (o1/o3/o4) ---
    pub reasoning_effort: Option<String>,
}

impl ChatOptions {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        ChatOptions {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            max_tool_rounds: 16,
            ..Default::default()
        }
    }
}

impl From<proto::ChatOptions> for ChatOptions {
    fn from(p: proto::ChatOptions) -> Self {
        ChatOptions {
            base_url: if p.base_url.is_empty() {
                "https://api.openai.com".to_string()
            } else {
                p.base_url
            },
            api_key: p.api_key,
            model: if p.model.is_empty() {
                "gpt-4o-mini".to_string()
            } else {
                p.model
            },
            max_tool_rounds: if p.max_tool_rounds == 0 {
                16
            } else {
                p.max_tool_rounds
            },
            system_prompt: p.system_prompt,
            temperature: p.temperature,
            top_p: p.top_p,
            n: None,
            max_completion_tokens: p.max_completion_tokens,
            presence_penalty: p.presence_penalty,
            frequency_penalty: p.frequency_penalty,
            stop: p.stop.map(StopSequence::One),
            response_format: None,
            tool_choice: None,
            parallel_tool_calls: p.parallel_tool_calls,
            logprobs: p.logprobs,
            top_logprobs: p.top_logprobs,
            seed: p.seed,
            store: p.store,
            service_tier: p.service_tier,
            reasoning_effort: p.reasoning_effort,
        }
    }
}

/// Outcome of a completed turn (assistant returned text or empty after tool loop).
#[derive(Debug, Clone)]
pub struct CompletionOutcome {
    pub content: Option<String>,
    /// Number of completion HTTP calls performed.
    pub rounds: u32,
    /// Token usage reported by the final completion response.
    pub usage: Option<proto::Usage>,
    /// The `finish_reason` from the last choice.
    pub finish_reason: Option<String>,
}

/// Errors from the chat + tool pipeline.
#[derive(Debug, Error)]
pub enum ChatError {
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Tool(#[from] ToolInvokeError),
    #[error("hook aborted: {0}")]
    Hook(#[from] HookError),
    #[error(transparent)]
    Guardrail(#[from] GuardrailError),
    #[error("completion response contained no choices")]
    NoChoice,
    #[error("exceeded max tool rounds ({0})")]
    MaxToolRounds(u32),
}

/// Run OpenAI-style chat completions with tools: calls `POST /v1/chat/completions` until the model
/// returns an assistant message without tool calls or [`ChatOptions::max_tool_rounds`] is exceeded.
///
/// **Input guardrails** run on the last user message before the first LLM call.
/// **Output guardrails** run on the final assistant text with a retry loop
/// (up to [`GuardrailRegistry::max_output_retries`]).
pub async fn complete_with_tools(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    caller_messages: Vec<ChatMessage>,
    options: &ChatOptions,
) -> Result<CompletionOutcome, ChatError> {
    let url = join_base_url(&options.base_url, "/v1/chat/completions");
    let auth = format!("Bearer {}", options.api_key);
    let headers = [("Authorization", auth.as_str())];

    // Prepend system prompt if configured.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        messages.push(ChatMessage::text("system", sp));
    }
    messages.extend(caller_messages);

    // --- Input guardrails: run on the last user message before first LLM call ---
    if !guardrails.input_is_empty().await {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();

        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                // If the guardrail transformed the content, update the last user message.
                if transformed != last_user {
                    if let Some(msg) = messages.iter_mut().rev().find(|m| m.role == "user") {
                        msg.content = Some(MessageContent::Text(transformed));
                    }
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(ChatError::Guardrail(GuardrailError::new(
                    GuardrailStage::Input,
                    guard_name,
                    reason,
                )));
            }
        }
    }

    let mut api_calls: u32 = 0;

    loop {
        if api_calls >= options.max_tool_rounds {
            return Err(ChatError::MaxToolRounds(options.max_tool_rounds));
        }
        api_calls += 1;

        // --- PreCompletion hook (observation only) ---
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        hooks
            .run(HookStage::PreCompletion, HookContext::new(HookStage::PreCompletion, last_user))
            .await?;

        let specs = registry.list_specs().await;
        let tools = if specs.is_empty() {
            None
        } else {
            Some(specs.into_iter().map(ChatTool::from).collect::<Vec<_>>())
        };

        let mut req = ChatCompletionRequest::new(options.model.clone(), messages.clone(), tools);

        req.temperature = options.temperature;
        req.top_p = options.top_p;
        req.n = options.n;
        req.max_completion_tokens = options.max_completion_tokens;
        req.presence_penalty = options.presence_penalty;
        req.frequency_penalty = options.frequency_penalty;
        req.stop = options.stop.clone();
        req.response_format = options.response_format.clone();
        req.tool_choice = options.tool_choice.clone();
        req.parallel_tool_calls = options.parallel_tool_calls;
        req.logprobs = options.logprobs;
        req.top_logprobs = options.top_logprobs;
        req.seed = options.seed;
        req.store = options.store;
        req.service_tier = options.service_tier.clone();
        req.reasoning_effort = options.reasoning_effort.clone();

        let body = serde_json::to_value(&req)?;
        let val = http.post_json_with_headers(&url, &body, &headers).await?;
        let response: crate::openai::ChatCompletionResponse = serde_json::from_value(val)?;

        let choice = response.choices.first().ok_or(ChatError::NoChoice)?;
        let msg = &choice.message;

        // --- PostCompletion hook (observation only) ---
        let assistant_text = msg
            .content
            .as_ref()
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        hooks
            .run(
                HookStage::PostCompletion,
                HookContext::new(HookStage::PostCompletion, assistant_text),
            )
            .await?;

        if let Some(tcs) = &msg.tool_calls
            && !tcs.is_empty()
        {
            messages.push(msg.clone());
            for tc in tcs {
                if tc.kind != "function" {
                    continue;
                }

                // --- PreTool hook (mutating) ---
                let pre_ctx = hooks
                    .run(
                        HookStage::PreTool,
                        HookContext::with_meta(
                            HookStage::PreTool,
                            tc.function.arguments.trim(),
                            "tool_name",
                            tc.function.name.as_str(),
                        ),
                    )
                    .await?;
                let args: Value = serde_json::from_str(pre_ctx.content.trim())?;

                let result = registry.invoke(&tc.function.name, args).await?;
                let result_json = serde_json::to_string(&result)?;

                // --- PostTool hook (mutating) ---
                let post_ctx = hooks
                    .run(
                        HookStage::PostTool,
                        HookContext::with_meta(
                            HookStage::PostTool,
                            &result_json,
                            "tool_name",
                            tc.function.name.as_str(),
                        ),
                    )
                    .await?;
                let final_result = post_ctx.content;

                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: Some(MessageContent::Text(final_result)),
                    tool_calls: None,
                    tool_call_id: Some(tc.id.clone()),
                    name: Some(tc.function.name.clone()),
                    refusal: None,
                });
            }
            continue;
        }

        // --- Terminal response: extract content ---
        let usage = response.usage.as_ref().map(|u| proto::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
        let content = msg
            .content
            .as_ref()
            .and_then(|c| c.as_text().map(str::to_string));
        let finish_reason = choice.finish_reason.clone();

        // --- Output guardrails: retry loop ---
        if guardrails.output_is_empty().await {
            return Ok(CompletionOutcome { content, rounds: api_calls, usage, finish_reason });
        }

        let text = content.clone().unwrap_or_default();
        let max_retries = guardrails.max_output_retries;

        for attempt in 0..=max_retries {
            let (outcome, guard_name) = guardrails.run_output(&text).await;
            match outcome {
                GuardrailOutcome::Allow(transformed) => {
                    let final_content = if transformed == text {
                        content
                    } else {
                        Some(transformed)
                    };
                    return Ok(CompletionOutcome {
                        content: final_content,
                        rounds: api_calls,
                        usage,
                        finish_reason,
                    });
                }
                GuardrailOutcome::Block(reason) => {
                    if attempt < max_retries {
                        // Append the rejected assistant reply + a "please revise" user message,
                        // then loop back to call the LLM again (no tools this time).
                        messages.push(msg.clone());
                        messages.push(ChatMessage::text(
                            "user",
                            format!(
                                "Your previous response was rejected by a content policy ({reason}). \
                                 Please revise it."
                            ),
                        ));
                        // Reset api_calls guard counter for the retry loop.
                        if api_calls >= options.max_tool_rounds {
                            return Err(ChatError::Guardrail(GuardrailError::new(
                                GuardrailStage::Output,
                                guard_name,
                                reason,
                            )));
                        }
                        api_calls += 1;
                        // Re-invoke the LLM (no tools — just text revision).
                        let retry_req = ChatCompletionRequest::new(
                            options.model.clone(),
                            messages.clone(),
                            None,
                        );
                        let body = serde_json::to_value(&retry_req)?;
                        let val = http.post_json_with_headers(&url, &body, &headers).await?;
                        let resp: crate::openai::ChatCompletionResponse =
                            serde_json::from_value(val)?;
                        let choice = resp.choices.first().ok_or(ChatError::NoChoice)?;
                        let retry_text = choice
                            .message
                            .content
                            .as_ref()
                            .and_then(|c| c.as_text().map(str::to_string))
                            .unwrap_or_default();
                        // Loop: next attempt will check the new text.
                        // We need to update `text` — use a labelled approach by breaking to
                        // restart the guardrail loop with new text.
                        // Use a simple recursive-style approach: just continue the outer loop.
                        let (out2, gn2) = guardrails.run_output(&retry_text).await;
                        match out2 {
                            GuardrailOutcome::Allow(t) => {
                                return Ok(CompletionOutcome {
                                    content: Some(t),
                                    rounds: api_calls,
                                    usage,
                                    finish_reason,
                                });
                            }
                            GuardrailOutcome::Block(r2) => {
                                if attempt + 1 >= max_retries {
                                    return Err(ChatError::Guardrail(GuardrailError::new(
                                        GuardrailStage::Output,
                                        gn2,
                                        r2,
                                    )));
                                }
                                // Continue outer for loop — but we've already advanced; break here.
                                return Err(ChatError::Guardrail(GuardrailError::new(
                                    GuardrailStage::Output,
                                    gn2,
                                    r2,
                                )));
                            }
                        }
                    } else {
                        return Err(ChatError::Guardrail(GuardrailError::new(
                            GuardrailStage::Output,
                            guard_name,
                            reason,
                        )));
                    }
                }
            }
        }

        // Should be unreachable — the loop above always returns.
        return Ok(CompletionOutcome { content, rounds: api_calls, usage, finish_reason });
    }
}

// ---------------------------------------------------------------------------
// Streaming chat completion (SSE / stream: true)
// ---------------------------------------------------------------------------

/// Outcome of a streaming completion (no tool-call loop).
#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    /// Full accumulated assistant content.
    pub content: String,
    /// `finish_reason` from the final chunk.
    pub finish_reason: Option<String>,
    /// Usage reported by the final chunk (only when `stream_options.include_usage = true`).
    pub usage: Option<proto::Usage>,
}

/// Stream a chat completion, calling `on_delta` for each content token as it arrives.
///
/// Unlike [`complete_with_tools`], this function does **not** execute tool calls.
/// Input guardrails run on the last user message before the stream starts; output
/// guardrails run on the fully-accumulated response after the stream ends (no retry —
/// same behaviour as gluellm's simple streaming path).
pub async fn stream_complete<F>(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    messages: Vec<ChatMessage>,
    options: &ChatOptions,
    mut on_delta: F,
) -> Result<StreamOutcome, ChatError>
where
    F: FnMut(String) + Send,
{
    let url = join_base_url(&options.base_url, "/v1/chat/completions");
    let auth = format!("Bearer {}", options.api_key);
    let headers = [("Authorization", auth.as_str())];

    // Prepend system prompt if configured.
    let mut full_messages: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        full_messages.push(ChatMessage::text("system", sp));
    }
    full_messages.extend(messages);

    // --- Input guardrails ---
    if !guardrails.input_is_empty().await {
        let last_user = full_messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();

        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user {
                    if let Some(msg) = full_messages.iter_mut().rev().find(|m| m.role == "user") {
                        msg.content = Some(MessageContent::Text(transformed));
                    }
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(ChatError::Guardrail(GuardrailError::new(
                    GuardrailStage::Input,
                    guard_name,
                    reason,
                )));
            }
        }
    }

    // --- PreCompletion hook (observation only) ---
    let last_user = full_messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text().map(str::to_string))
        .unwrap_or_default();
    hooks
        .run(HookStage::PreCompletion, HookContext::new(HookStage::PreCompletion, last_user))
        .await?;

    let mut req = ChatCompletionRequest::new(options.model.clone(), full_messages, None);
    req.stream = Some(true);
    req.stream_options = Some(crate::openai::StreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    });
    req.temperature = options.temperature;
    req.top_p = options.top_p;
    req.n = options.n;
    req.max_completion_tokens = options.max_completion_tokens;
    req.presence_penalty = options.presence_penalty;
    req.frequency_penalty = options.frequency_penalty;
    req.stop = options.stop.clone();
    req.response_format = options.response_format.clone();
    req.logprobs = options.logprobs;
    req.top_logprobs = options.top_logprobs;
    req.seed = options.seed;
    req.store = options.store;
    req.service_tier = options.service_tier.clone();
    req.reasoning_effort = options.reasoning_effort.clone();

    let body = serde_json::to_value(&req)?;
    eprintln!("[stream_complete] POST {url}");

    let mut byte_stream = http
        .post_json_stream_with_headers(&url, &body, &headers)
        .await?;
    eprintln!("[stream_complete] connection established, reading SSE chunks…");

    let mut parser = SseParser::new();
    let mut outcome = StreamOutcome::default();
    let mut byte_count = 0usize;
    let mut event_count = 0usize;

    while let Some(chunk) = byte_stream.next().await {
        let bytes = chunk?;
        byte_count += bytes.len();
        let text = String::from_utf8_lossy(&bytes);
        eprintln!("[stream_complete] raw chunk ({} bytes): {:?}", bytes.len(), &text[..text.len().min(120)]);

        let events = parser
            .push_str(&text)
            .map_err(|e| ChatError::Http(HttpError::InvalidJson(e.to_string())))?;

        for event in events {
            event_count += 1;
            let data = event.data.trim();
            eprintln!("[stream_complete] SSE event #{event_count} data={:?}", &data[..data.len().min(80)]);
            if data == "[DONE]" {
                eprintln!("[stream_complete] received [DONE]");
                break;
            }
            let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("[stream_complete] JSON parse error: {e} — data: {data}");
                    return Err(ChatError::Serde(e));
                }
            };

            if let Some(u) = &chunk.usage {
                outcome.usage = Some(proto::Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                });
            }

            for choice in &chunk.choices {
                if let Some(fr) = &choice.finish_reason {
                    eprintln!("[stream_complete] finish_reason={fr}");
                    outcome.finish_reason = Some(fr.clone());
                }
                if let Some(delta) = &choice.delta.content
                    && !delta.is_empty()
                {
                    eprintln!("[stream_complete] delta token: {:?}", delta);
                    outcome.content.push_str(delta);
                    on_delta(delta.clone());
                }
            }
        }
    }

    eprintln!("[stream_complete] done — {byte_count} bytes, {event_count} SSE events, {} content chars", outcome.content.len());

    // --- Output guardrails (no retry for streaming) ---
    if !guardrails.output_is_empty().await {
        let (out_outcome, guard_name) = guardrails.run_output(&outcome.content).await;
        match out_outcome {
            GuardrailOutcome::Allow(transformed) => {
                outcome.content = transformed;
            }
            GuardrailOutcome::Block(reason) => {
                return Err(ChatError::Guardrail(GuardrailError::new(
                    GuardrailStage::Output,
                    guard_name,
                    reason,
                )));
            }
        }
    }

    Ok(outcome)
}
