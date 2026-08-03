//! OpenAI Responses API (`POST /v1/responses`) with tool loop and `previous_response_id` threading.

use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tracing::instrument;

use crate::chat::{
    ChatError, ChatOptions, StreamToolOutcome, condense_tool_round,
    conversation_messages_for_client, credentials_for, dispatch_one, estimate_context_chars,
    fail_partial, maybe_summarize_messages, new_tool_loop_guard, notify_llm_payload,
    observation_hook_ctx, post_json_with_model_fallback, stream_tools,
};
use crate::costing::estimate_model_call_cost_usd;
use crate::events::{ProcessEvent, ProcessEventKind, StatusEmitter, emit_safe};
use crate::guardrails::{GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage};
use crate::hooks::{HookRegistry, HookStage};
use crate::http::{HttpClient, join_base_url, sse::SseParser};
use crate::openai::ChatCompletionChunk;
use crate::openai::{
    ChatMessage, ContentPart, FunctionCall, ImageDetail, MessageContent, ToolCall, ToolChoice,
};
use crate::proto;
use crate::tools::{ToolRegistry, ToolSpec};

impl From<ResponseError> for ChatError {
    fn from(e: ResponseError) -> Self {
        match e {
            ResponseError::Chat(c) => c,
            ResponseError::Http(h) => ChatError::Http(h),
            ResponseError::Serde(s) => ChatError::Serde(s),
            ResponseError::NoOutput => ChatError::NoChoice,
            ResponseError::MaxToolRounds(n) => ChatError::MaxToolRounds(n),
            ResponseError::Cancelled => ChatError::Cancelled,
            ResponseError::StreamFailed(msg) => ChatError::Api(msg),
        }
    }
}

/// Errors from the Responses API pipeline.
#[derive(Debug, Error)]
pub enum ResponseError {
    #[error(transparent)]
    Chat(ChatError),
    #[error(transparent)]
    Http(#[from] crate::http::Error),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error("response contained no output")]
    NoOutput,
    #[error("exceeded max tool rounds ({0})")]
    MaxToolRounds(u32),
    #[error("request cancelled")]
    Cancelled,
    #[error("response stream failed: {0}")]
    StreamFailed(String),
}

impl From<ChatError> for ResponseError {
    fn from(e: ChatError) -> ResponseError {
        match e {
            ChatError::Cancelled => ResponseError::Cancelled,
            other => ResponseError::Chat(other),
        }
    }
}

impl From<crate::hooks::HookError> for ResponseError {
    fn from(e: crate::hooks::HookError) -> Self {
        ResponseError::Chat(ChatError::Hook(e))
    }
}

fn fail_response_partial(
    cause: ResponseError,
    messages: &[ChatMessage],
    had_system_prompt: bool,
) -> ResponseError {
    let chat = match cause {
        ResponseError::Chat(c) => c,
        ResponseError::Http(h) => ChatError::Http(h),
        ResponseError::Serde(s) => ChatError::Serde(s),
        ResponseError::NoOutput => ChatError::NoChoice,
        ResponseError::MaxToolRounds(n) => ChatError::MaxToolRounds(n),
        ResponseError::Cancelled => ChatError::Cancelled,
        ResponseError::StreamFailed(msg) => ChatError::Api(msg),
    };
    ResponseError::Chat(fail_partial(chat, messages, had_system_prompt))
}

/// Completed Responses API turn.
#[derive(Debug, Clone)]
pub struct ResponseOutcome {
    pub id: String,
    pub content: Option<String>,
    pub rounds: u32,
    pub usage: Option<proto::Usage>,
    pub request_id: String,
    pub model_used: String,
    pub raw_output: Vec<ResponseOutputItem>,
}

/// Streaming Responses API outcome.
#[derive(Debug, Clone)]
pub struct ResponseStreamOutcome {
    pub id: String,
    pub content: String,
    pub usage: Option<proto::Usage>,
    pub request_id: String,
    pub model_used: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message {
        role: String,
        #[serde(default)]
        content: Value,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message {
        id: Option<String>,
        role: Option<String>,
        content: Option<Vec<OutputContentPart>>,
    },
    FunctionCall {
        /// Responses output item id (`fc_...`) when the API also sends `call_id`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default)]
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseReasoning {
    effort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseCreateRequest {
    model: String,
    input: ResponseInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponseTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponseReasoning>,
}

fn reasoning_from_options(options: &ChatOptions) -> Option<ResponseReasoning> {
    options
        .reasoning_effort
        .as_ref()
        .map(|effort| ResponseReasoning {
            effort: effort.clone(),
            summary: options.reasoning_summary.api_value().map(str::to_string),
        })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseTool {
    r#type: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponseObject {
    #[serde(default)]
    id: String,
    #[serde(default)]
    output: Vec<ResponseOutputItem>,
    usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

fn tools_from_registry(specs: Vec<ToolSpec>) -> Vec<ResponseTool> {
    specs
        .into_iter()
        .map(|s| ResponseTool {
            r#type: "function".to_string(),
            name: s.name,
            description: s.description,
            parameters: s.parameters_schema,
        })
        .collect()
}

fn usage_from_response(u: &ResponseUsage) -> proto::Usage {
    proto::Usage {
        prompt_tokens: u.input_tokens.unwrap_or(0),
        completion_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.total_tokens.unwrap_or(0),
        cached_tokens: None,
        reasoning_tokens: None,
    }
}

fn extract_output_text(output: &[ResponseOutputItem]) -> Option<String> {
    let mut parts = Vec::new();
    for item in output {
        if let ResponseOutputItem::Message { content, .. } = item {
            if let Some(content_parts) = content {
                for part in content_parts {
                    if let OutputContentPart::OutputText { text } = part {
                        parts.push(text.clone());
                    }
                }
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(""))
    }
}

fn effective_function_call_id(call_id: &str, id: &Option<String>) -> String {
    if !call_id.is_empty() {
        call_id.to_string()
    } else {
        id.clone().unwrap_or_default()
    }
}

fn looks_like_function_call_id(id: &str) -> bool {
    id.starts_with("call_")
}

/// Prefer completed output when it has real `call_…` ids; otherwise keep streamed ids
/// (completed items sometimes only carry `fc_…` output item ids).
fn resolve_round_function_calls(round_state: &ResponsesStreamRound) -> Vec<StreamedFunctionCall> {
    let from_output = function_calls_from_output(&round_state.raw_output);
    if from_output.is_empty() {
        return round_state.function_calls.clone();
    }
    let output_has_call_ids = from_output
        .iter()
        .any(|c| looks_like_function_call_id(&c.call_id));
    if output_has_call_ids {
        return from_output;
    }
    let stream_has_call_ids = round_state
        .function_calls
        .iter()
        .any(|c| looks_like_function_call_id(&c.call_id));
    if stream_has_call_ids {
        return round_state.function_calls.clone();
    }
    from_output
}

/// `call_id` from a Responses SSE payload (`call_...`), not the output item id (`fc_...`).
fn explicit_call_id_from_value(v: &Value) -> Option<String> {
    v.get("call_id")
        .or_else(|| v.pointer("/item/call_id"))
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn push_streamed_function_call(
    round_state: &mut ResponsesStreamRound,
    call: StreamedFunctionCall,
) -> bool {
    if round_state
        .function_calls
        .iter()
        .any(|c| c.call_id == call.call_id)
    {
        return false;
    }
    round_state.function_calls.push(call);
    true
}

fn function_calls(output: &[ResponseOutputItem]) -> Vec<(String, String, String)> {
    output
        .iter()
        .filter_map(|item| {
            if let ResponseOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } = item
            {
                let resolved = effective_function_call_id(call_id, id);
                Some((resolved, name.clone(), arguments.clone()))
            } else {
                None
            }
        })
        .collect()
}

fn responses_rate_limit_key(options: &ChatOptions) -> Option<crate::providers::RateLimitKey> {
    let model_ref = crate::providers::parse_model_ref(&options.model);
    if model_ref.provider != crate::providers::ProviderId::OpenAi {
        return None;
    }
    let creds = credentials_for(options);
    crate::providers::rate_limit_key_for(&model_ref, &creds).ok()
}

async fn post_response(
    http: &HttpClient,
    url: &str,
    body: &Value,
    headers: &[(&str, &str)],
    options: &ChatOptions,
    request_id: &str,
    round: u32,
) -> Result<(ResponseObject, String), ResponseError> {
    let (val, model) = post_json_with_model_fallback(
        http,
        url,
        body,
        headers,
        options,
        request_id,
        round,
        responses_rate_limit_key(options),
    )
    .await?;
    let resp: ResponseObject = serde_json::from_value(val)?;
    Ok((resp, model))
}

/// Non-streaming Responses API tool loop with `previous_response_id` threading.
#[instrument(
    skip(http, registry, hooks, guardrails, options, user_message),
    fields(model = %options.model)
)]
pub async fn complete_with_tools(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    user_message: impl Into<String>,
    options: &ChatOptions,
) -> Result<ResponseOutcome, ResponseError> {
    let user_message = user_message.into();
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let url = join_base_url(&options.base_url, "/v1/responses");
    let auth = format!("Bearer {}", options.api_key.expose_secret());
    let headers = [("Authorization", auth.as_str())];

    // Input guardrails on user message
    if !guardrails.input_is_empty().await {
        let (outcome, guard_name) = guardrails.run_input(&user_message).await;
        match outcome {
            GuardrailOutcome::Allow(_) => {}
            GuardrailOutcome::Block(reason) => {
                return Err(ResponseError::Chat(ChatError::Guardrail(
                    GuardrailError::new(GuardrailStage::Input, guard_name, reason),
                )));
            }
        }
    }

    let specs = registry.list_specs().await;
    let tools = if specs.is_empty() {
        None
    } else {
        Some(tools_from_registry(specs))
    };

    let mut api_calls = 0u32;
    let mut model_used;
    let mut previous_response_id: Option<String> = None;
    let mut tool_input: Option<Vec<ResponseInputItem>> = None;
    let loop_guard = new_tool_loop_guard();

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            return Err(ResponseError::MaxToolRounds(options.max_tool_rounds));
        }
        api_calls += 1;
        if let Some(logger) = &options.context_event_logger {
            logger.log(format!(
                "context responses_round={} user_chars={} tool_items={}",
                api_calls,
                user_message.chars().count(),
                tool_input.as_ref().map_or(0, Vec::len),
            ));
        }

        hooks
            .run(
                HookStage::PreCompletion,
                observation_hook_ctx(
                    HookStage::PreCompletion,
                    user_message.clone(),
                    &request_id,
                    api_calls,
                    &options.model,
                ),
            )
            .await?;

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        notify_llm_payload(
            options,
            api_calls,
            &request_id,
            &[ChatMessage::text("user", user_message.clone())],
        );

        let input = if let Some(items) = &tool_input {
            ResponseInput::Items(items.clone())
        } else {
            ResponseInput::Text(user_message.clone())
        };

        let req = ResponseCreateRequest {
            model: options.model.clone(),
            input,
            instructions: options.system_prompt.clone(),
            tools: tools.clone(),
            stream: None,
            previous_response_id: previous_response_id.clone(),
            tool_choice: options.tool_choice.clone(),
            reasoning: reasoning_from_options(options),
        };

        let mut body = serde_json::to_value(&req)?;
        if let Some(extra) = &options.extra_json {
            if let (Value::Object(b), Value::Object(e)) = (&mut body, extra) {
                for (k, v) in e {
                    b.insert(k.clone(), v.clone());
                }
            }
        }

        let (resp, round_model) =
            post_response(http, &url, &body, &headers, options, &request_id, api_calls).await?;

        model_used = round_model;
        previous_response_id = Some(resp.id.clone());

        let usage_proto = resp.usage.as_ref().map(usage_from_response);
        let estimated_cost = usage_proto
            .as_ref()
            .map(|u| estimate_model_call_cost_usd(&model_used, u));

        let calls = function_calls(&resp.output);
        let tool_call_count = calls.len() as u32;
        emit_safe(options.status_emitter.as_ref(), {
            let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
            ev.round = api_calls;
            ev.tool_call_count = tool_call_count;
            ev.usage = usage_proto.clone();
            ev.estimated_cost_usd = estimated_cost;
            ev
        })
        .await;

        if !calls.is_empty() {
            let tool_calls: Vec<ToolCall> = calls
                .iter()
                .map(|(call_id, name, args)| ToolCall {
                    id: call_id.clone(),
                    kind: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: args.clone(),
                    },
                })
                .collect();
            let results = futures_util::future::join_all(tool_calls.iter().map(|tc| {
                dispatch_one(
                    tc,
                    hooks,
                    registry,
                    options.status_emitter.as_ref(),
                    &request_id,
                    api_calls,
                    &model_used,
                    true,
                    options.tool_result_max_chars,
                    Some(&loop_guard),
                )
            }))
            .await;

            let mut outputs = Vec::new();
            for (r, (call_id, _, _)) in results.into_iter().zip(calls.iter()) {
                let msg = r?;
                let out_text = msg
                    .content
                    .as_ref()
                    .and_then(|c| c.as_text())
                    .unwrap_or("{}");
                outputs.push(ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: out_text.to_string(),
                });
            }
            tool_input = Some(outputs);
            continue;
        }

        let content = extract_output_text(&resp.output);

        if !guardrails.output_is_empty().await {
            let text = content.clone().unwrap_or_default();
            let (guard_out, guard_name) = guardrails.run_output(&text).await;
            match guard_out {
                GuardrailOutcome::Allow(transformed) => {
                    break ResponseOutcome {
                        id: resp.id,
                        content: Some(transformed),
                        rounds: api_calls,
                        usage: usage_proto,
                        request_id: request_id.clone(),
                        model_used: model_used.clone(),
                        raw_output: resp.output,
                    };
                }
                GuardrailOutcome::Block(reason) => {
                    return Err(ResponseError::Chat(ChatError::Guardrail(
                        GuardrailError::new(GuardrailStage::Output, guard_name, reason),
                    )));
                }
            }
        }

        break ResponseOutcome {
            id: resp.id,
            content,
            rounds: api_calls,
            usage: usage_proto,
            request_id: request_id.clone(),
            model_used: model_used.clone(),
            raw_output: resp.output,
        };
    };

    tracing::info!(
        request_id = %outcome.request_id,
        model = %outcome.model_used,
        rounds = outcome.rounds,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "responses complete_with_tools finished"
    );

    Ok(outcome)
}

/// Stream Responses API output; `on_delta` receives text token deltas.
pub async fn stream_response<F>(
    http: &HttpClient,
    _hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    user_message: impl Into<String>,
    options: &ChatOptions,
    mut on_delta: F,
) -> Result<ResponseStreamOutcome, ResponseError>
where
    F: FnMut(String) + Send,
{
    let user_message = user_message.into();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    if !guardrails.input_is_empty().await {
        let (outcome, guard_name) = guardrails.run_input(&user_message).await;
        if let GuardrailOutcome::Block(reason) = outcome {
            return Err(ResponseError::Chat(ChatError::Guardrail(
                GuardrailError::new(GuardrailStage::Input, guard_name, reason),
            )));
        }
    }

    let url = join_base_url(&options.base_url, "/v1/responses");
    let auth = format!("Bearer {}", options.api_key.expose_secret());
    let headers = [("Authorization", auth.as_str())];

    let req = ResponseCreateRequest {
        model: options.model.clone(),
        input: ResponseInput::Text(user_message),
        instructions: options.system_prompt.clone(),
        tools: None,
        stream: Some(true),
        previous_response_id: None,
        tool_choice: options.tool_choice.clone(),
        reasoning: reasoning_from_options(options),
    };

    let body = serde_json::to_value(&req)?;
    let stream = http
        .post_json_stream_with_headers(&url, &body, &headers, responses_rate_limit_key(options))
        .await?;

    let mut parser = SseParser::new();
    let mut content = String::new();
    let mut response_id = String::new();
    let model_used = options.model.clone();
    let mut usage: Option<proto::Usage> = None;
    let mut stream_error: Option<String> = None;

    futures_util::pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        let text = String::from_utf8_lossy(&bytes);
        let events = parser.push_str(&text).map_err(ResponseError::Http)?;
        for event in events {
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(data).unwrap_or(Value::Null);
            let event_type = resolve_sse_event_type(event.event.as_deref(), &v);
            match event_type.as_str() {
                "response.created" | "response.in_progress" => {
                    if let Some(id) = v
                        .pointer("/response/id")
                        .or_else(|| v.get("id"))
                        .and_then(|id| id.as_str())
                    {
                        response_id = id.to_string();
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = v
                        .get("delta")
                        .and_then(|d| d.as_str())
                        .or_else(|| v.pointer("/delta/text").and_then(|t| t.as_str()))
                    {
                        content.push_str(delta);
                        on_delta(delta.to_string());
                    }
                }
                "response.output_text.done" => {
                    if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                        if content.is_empty() {
                            on_delta(text.to_string());
                        }
                        content = text.to_string();
                    }
                }
                "response.content_part.done" => {
                    let part_type = v
                        .pointer("/part/type")
                        .and_then(|t| t.as_str())
                        .unwrap_or_default();
                    if part_type == "output_text"
                        && let Some(text) = v.pointer("/part/text").and_then(|t| t.as_str())
                    {
                        if content.is_empty() {
                            on_delta(text.to_string());
                        }
                        content = text.to_string();
                    }
                }
                "response.completed" => {
                    if let Some(u) = v.pointer("/response/usage") {
                        usage = Some(proto::Usage {
                            prompt_tokens: u
                                .get("input_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0) as u32,
                            completion_tokens: u
                                .get("output_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0) as u32,
                            total_tokens: u
                                .get("total_tokens")
                                .and_then(|x| x.as_u64())
                                .unwrap_or(0) as u32,
                            cached_tokens: None,
                            reasoning_tokens: None,
                        });
                    }
                    if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                        response_id = id.to_string();
                    }
                }
                "error" | "response.error" | "response.failed" => {
                    if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                        response_id = id.to_string();
                    }
                    if let Some(msg) = extract_sse_error_message(&v) {
                        tracing::warn!(event_type = %event_type, error = %msg, "responses: stream error event");
                        stream_error = Some(msg);
                    } else {
                        stream_error = Some(format!("{event_type} (no details)"));
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(err) = stream_error {
        emit_llm_call_error(options.status_emitter.as_ref(), &request_id, &model_used, 1);
        return Err(ResponseError::StreamFailed(err));
    }

    if !guardrails.output_is_empty().await {
        let (outcome, guard_name) = guardrails.run_output(&content).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => content = transformed,
            GuardrailOutcome::Block(reason) => {
                return Err(ResponseError::Chat(ChatError::Guardrail(
                    GuardrailError::new(GuardrailStage::Output, guard_name, reason),
                )));
            }
        }
    }

    Ok(ResponseStreamOutcome {
        id: response_id,
        content,
        usage,
        request_id,
        model_used,
    })
}

/// Map chat history (excluding system messages) to Responses API input items.
///
/// Drops orphan `tool` / `function_call_output` items whose `call_id` is not present
/// on a preceding assistant `function_call` — the API rejects those with
/// `No tool call found for function call output`.
#[must_use]
pub(crate) fn chat_messages_to_response_input(messages: &[ChatMessage]) -> ResponseInput {
    let mut items = Vec::new();
    let mut known_call_ids = std::collections::HashSet::new();
    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        if msg.role == "tool" {
            if let Some(call_id) = &msg.tool_call_id {
                if !known_call_ids.contains(call_id) {
                    tracing::warn!(
                        call_id,
                        "responses: dropping orphan tool result (no matching function_call)"
                    );
                    continue;
                }
                let output = msg
                    .content
                    .as_ref()
                    .map(MessageContent::text_for_summary)
                    .unwrap_or_default();
                items.push(ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output,
                });
            }
            continue;
        }
        if msg.role == "assistant" {
            if let Some(content) = msg
                .content
                .as_ref()
                .and_then(message_content_to_response_value)
            {
                items.push(ResponseInputItem::Message {
                    role: "assistant".into(),
                    content,
                });
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    if tc.kind == "function" {
                        known_call_ids.insert(tc.id.clone());
                        items.push(ResponseInputItem::FunctionCall {
                            call_id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        });
                    }
                }
            }
            continue;
        }
        if let Some(content) = msg
            .content
            .as_ref()
            .and_then(message_content_to_response_value)
        {
            items.push(ResponseInputItem::Message {
                role: msg.role.clone(),
                content,
            });
        }
    }
    ResponseInput::Items(items)
}

fn message_content_to_response_value(content: &MessageContent) -> Option<Value> {
    match content {
        MessageContent::Text(text) if !text.is_empty() => Some(Value::String(text.clone())),
        MessageContent::Text(_) => None,
        MessageContent::Parts(parts) => {
            let blocks = content_parts_to_response_parts(parts);
            if blocks.is_empty() {
                None
            } else if blocks.len() == 1
                && blocks[0].get("type").and_then(Value::as_str) == Some("input_text")
                && let Some(text) = blocks[0].get("text").and_then(Value::as_str)
            {
                Some(Value::String(text.to_string()))
            } else {
                Some(Value::Array(blocks))
            }
        }
    }
}

fn content_parts_to_response_parts(parts: &[ContentPart]) -> Vec<Value> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text } if !text.is_empty() => {
                out.push(json!({"type": "input_text", "text": text}));
            }
            ContentPart::ImageUrl { image_url } => {
                let mut block = json!({
                    "type": "input_image",
                    "image_url": image_url.url,
                });
                if let Some(detail) = &image_url.detail {
                    block["detail"] = json!(image_detail_to_api(detail));
                }
                out.push(block);
            }
            ContentPart::ImageRef { hash, filename } => {
                let label = filename.as_deref().unwrap_or(hash.as_str());
                out.push(json!({
                    "type": "input_text",
                    "text": format!("[missing image: {label}]"),
                }));
            }
            ContentPart::File { file } => {
                if let Some(data) = &file.file_data {
                    let mut block = json!({
                        "type": "input_file",
                        "file_data": data,
                    });
                    if let Some(name) = &file.filename {
                        block["filename"] = json!(name);
                    }
                    out.push(block);
                }
            }
            ContentPart::InputAudio { .. } => {}
            ContentPart::Text { .. } => {}
        }
    }
    out
}

fn image_detail_to_api(detail: &ImageDetail) -> &'static str {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
    }
}

/// Build the next Responses `input`: chain tool outputs only when `previous_response_id`
/// is usable; otherwise send full sanitized history.
fn next_response_input(
    messages: &[ChatMessage],
    previous_response_id: &Option<String>,
    tool_input: &Option<Vec<ResponseInputItem>>,
) -> ResponseInput {
    let can_chain = previous_response_id
        .as_ref()
        .is_some_and(|id| !id.is_empty())
        && tool_input.as_ref().is_some_and(|items| !items.is_empty());
    if can_chain {
        return ResponseInput::Items(tool_input.clone().unwrap_or_default());
    }
    if tool_input.is_some() {
        tracing::warn!(
            "responses: refusing tool-output chaining without previous_response_id; using full history"
        );
    }
    chat_messages_to_response_input(messages)
}

#[derive(Debug, Clone)]
struct StreamedFunctionCall {
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct ResponsesStreamRound {
    reasoning: String,
    content: String,
    function_calls: Vec<StreamedFunctionCall>,
    response_id: String,
    usage: Option<proto::Usage>,
    raw_output: Vec<ResponseOutputItem>,
    stream_error: Option<String>,
    chat_tool_dispatch: stream_tools::StreamingToolDispatch,
    chat_finish_reason: Option<String>,
}

fn format_sse_error_object(err: &Value) -> Option<String> {
    let code = err
        .get("code")
        .or_else(|| err.get("type"))
        .and_then(|c| c.as_str())
        .filter(|c| !c.is_empty());
    let message = err
        .get("message")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty());
    match (code, message) {
        (Some(c), Some(m)) => Some(format!("{c}: {m}")),
        (Some(c), None) => Some(c.to_string()),
        (None, Some(m)) => Some(m.to_string()),
        _ => None,
    }
}

fn extract_sse_error_message(v: &Value) -> Option<String> {
    if let Some(err) = v
        .pointer("/response/error")
        .and_then(format_sse_error_object)
    {
        return Some(err);
    }
    if let Some(err) = v.get("error").and_then(format_sse_error_object) {
        return Some(err);
    }
    v.get("message")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .map(str::to_string)
}

fn extract_stream_delta(v: &Value) -> Option<String> {
    v.get("delta")
        .and_then(|d| d.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.pointer("/delta/text")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
        .or_else(|| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .or_else(|| {
            v.pointer("/part/text")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
}

fn extract_reasoning_summaries_from_output(output: &Value) -> Option<String> {
    let arr = output.as_array()?;
    let mut parts = Vec::new();
    for item in arr {
        if item.get("type").and_then(|t| t.as_str()) != Some("reasoning") {
            continue;
        }
        if let Some(summary) = item.get("summary").and_then(|s| s.as_array()) {
            for part in summary {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn function_calls_from_output(output: &[ResponseOutputItem]) -> Vec<StreamedFunctionCall> {
    output
        .iter()
        .filter_map(|item| {
            if let ResponseOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } = item
            {
                Some(StreamedFunctionCall {
                    call_id: effective_function_call_id(call_id, id),
                    name: name.clone(),
                    arguments: arguments.clone(),
                })
            } else {
                None
            }
        })
        .collect()
}

fn resolve_sse_event_type(sse_event: Option<&str>, v: &Value) -> String {
    if let Some(t) = v
        .get("type")
        .and_then(|t| t.as_str())
        .filter(|t| !t.is_empty())
    {
        return t.to_string();
    }
    sse_event
        .filter(|e| !e.is_empty())
        .unwrap_or_default()
        .to_string()
}

fn deliver_output_text_if_needed(
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
    text: String,
) {
    if text.is_empty() {
        return;
    }
    let was_empty = round_state.content.is_empty();
    round_state.content = text.clone();
    if was_empty {
        on_delta(text);
    }
}

fn parse_response_output_items(output: &Value) -> Vec<ResponseOutputItem> {
    let Some(arr) = output.as_array() else {
        return Vec::new();
    };
    if arr.is_empty() {
        return Vec::new();
    }
    if let Ok(parsed) = serde_json::from_value::<Vec<ResponseOutputItem>>(Value::Array(arr.clone()))
    {
        return parsed;
    }
    tracing::debug!("responses: batch output parse failed, falling back per-item");
    arr.iter()
        .filter_map(|item| serde_json::from_value::<ResponseOutputItem>(item.clone()).ok())
        .collect()
}

async fn ingest_stream_output_item(
    parsed: ResponseOutputItem,
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
    status_emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    round: u32,
    model: &str,
) {
    round_state.raw_output.push(parsed.clone());
    match parsed {
        ResponseOutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
        } if !name.is_empty() => {
            let resolved_call_id = effective_function_call_id(&call_id, &id);
            let call = StreamedFunctionCall {
                call_id: resolved_call_id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            };
            if push_streamed_function_call(round_state, call)
                && let Some(emitter) = status_emitter
            {
                let mut ev = ProcessEvent::new(ProcessEventKind::ToolCallStart, request_id, model);
                ev.round = round;
                ev.metadata.insert("tool_name".to_string(), name);
                ev.metadata.insert(
                    "arguments".to_string(),
                    crate::chat::truncate_tool_event_metadata(arguments.trim()),
                );
                emit_safe(Some(emitter), ev).await;
            }
        }
        item => {
            if round_state.content.is_empty()
                && let Some(text) = extract_output_text(&[item])
            {
                round_state.content = text.clone();
                on_delta(text);
            }
        }
    }
}

fn merge_chat_tool_dispatch(round_state: &mut ResponsesStreamRound) {
    if round_state.function_calls.is_empty() {
        for tc in round_state.chat_tool_dispatch.finish_remaining() {
            if tc.function.name.is_empty() {
                continue;
            }
            round_state.function_calls.push(StreamedFunctionCall {
                call_id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments,
            });
        }
    }
}

fn apply_chat_completion_chunk_event(
    v: &Value,
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
) -> bool {
    if !v.get("choices").is_some_and(|choices| choices.is_array()) {
        return false;
    }
    let Ok(chunk) = serde_json::from_value::<ChatCompletionChunk>(v.clone()) else {
        return false;
    };
    if !chunk.id.is_empty() {
        round_state.response_id = chunk.id.clone();
    }
    let prev_len = round_state.content.len();
    let _ready = stream_tools::apply_openai_chunk(
        &chunk,
        &mut round_state.content,
        &mut round_state.chat_tool_dispatch,
        &mut round_state.chat_finish_reason,
        &mut round_state.usage,
    );
    if round_state.content.len() > prev_len {
        on_delta(round_state.content[prev_len..].to_string());
    }
    true
}

async fn apply_responses_stream_event(
    v: &Value,
    event_type: &str,
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
    on_reasoning_delta: &mut impl FnMut(String),
    status_emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    round: u32,
    model: &str,
) {
    match event_type {
        "response.created" | "response.in_progress" => {
            if let Some(id) = v
                .pointer("/response/id")
                .or_else(|| v.get("id"))
                .and_then(|id| id.as_str())
            {
                round_state.response_id = id.to_string();
            }
        }
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            if let Some(delta) = extract_stream_delta(v) {
                round_state.reasoning.push_str(&delta);
                on_reasoning_delta(delta.clone());
                if let Some(emitter) = status_emitter {
                    let mut ev =
                        ProcessEvent::new(ProcessEventKind::ReasoningDelta, request_id, model);
                    ev.round = round;
                    ev.metadata.insert("delta".to_string(), delta);
                    emit_safe(Some(emitter), ev).await;
                }
            }
        }
        "response.output_text.delta" => {
            if let Some(delta) = extract_stream_delta(v) {
                round_state.content.push_str(&delta);
                on_delta(delta);
            }
        }
        "response.output_text.done" => {
            if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                deliver_output_text_if_needed(round_state, on_delta, text.to_string());
            }
        }
        "response.content_part.done" => {
            let part_type = v
                .pointer("/part/type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            if part_type == "output_text"
                && let Some(text) = v.pointer("/part/text").and_then(|t| t.as_str())
            {
                deliver_output_text_if_needed(round_state, on_delta, text.to_string());
            }
        }
        "response.function_call_arguments.done" => {
            // Only trust explicit `call_id` here. `item_id` / `id` are `fc_...` output item ids
            // and must not be sent back as `function_call_output.call_id`.
            if let Some(call_id) = explicit_call_id_from_value(v) {
                let name = v
                    .get("name")
                    .or_else(|| v.pointer("/item/name"))
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = v
                    .get("arguments")
                    .or_else(|| v.pointer("/item/arguments"))
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string();
                if !name.is_empty() {
                    let call = StreamedFunctionCall {
                        call_id,
                        name: name.clone(),
                        arguments: arguments.clone(),
                    };
                    if push_streamed_function_call(round_state, call)
                        && let Some(emitter) = status_emitter
                    {
                        let mut ev =
                            ProcessEvent::new(ProcessEventKind::ToolCallStart, request_id, model);
                        ev.round = round;
                        ev.metadata.insert("tool_name".to_string(), name);
                        ev.metadata.insert(
                            "arguments".to_string(),
                            crate::chat::truncate_tool_event_metadata(arguments.trim()),
                        );
                        emit_safe(Some(emitter), ev).await;
                    }
                }
            }
        }
        "response.output_item.done" => {
            if let Some(item) = v.get("item") {
                if let Ok(parsed) = serde_json::from_value::<ResponseOutputItem>(item.clone()) {
                    ingest_stream_output_item(
                        parsed,
                        round_state,
                        on_delta,
                        status_emitter,
                        request_id,
                        round,
                        model,
                    )
                    .await;
                } else {
                    tracing::debug!("responses: failed to parse output_item.done item");
                }
            }
        }
        "response.completed" => {
            if let Some(u) = v.pointer("/response/usage") {
                round_state.usage = Some(proto::Usage {
                    prompt_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                        as u32,
                    completion_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                        as u32,
                    total_tokens: u.get("total_tokens").and_then(|x| x.as_u64()).unwrap_or(0)
                        as u32,
                    cached_tokens: None,
                    reasoning_tokens: None,
                });
            }
            if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                round_state.response_id = id.to_string();
            }
            if let Some(output) = v.pointer("/response/output") {
                let parsed = parse_response_output_items(output);
                if !parsed.is_empty() {
                    round_state.raw_output = parsed;
                } else if output.as_array().is_some_and(|a| !a.is_empty()) {
                    tracing::debug!(
                        "responses: response.completed output present but no items parsed"
                    );
                }
                if round_state.reasoning.is_empty()
                    && let Some(text) = extract_reasoning_summaries_from_output(output)
                {
                    round_state.reasoning = text.clone();
                    on_reasoning_delta(text.clone());
                    if let Some(emitter) = status_emitter {
                        let mut ev =
                            ProcessEvent::new(ProcessEventKind::ReasoningDelta, request_id, model);
                        ev.round = round;
                        ev.metadata.insert("delta".to_string(), text);
                        emit_safe(Some(emitter), ev).await;
                    }
                }
                if round_state.content.is_empty()
                    && let Some(text) = extract_output_text(&round_state.raw_output)
                {
                    round_state.content = text.clone();
                    on_delta(text);
                }
            }
        }
        "error" | "response.error" | "response.failed" => {
            if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                round_state.response_id = id.to_string();
            }
            if let Some(msg) = extract_sse_error_message(v) {
                tracing::warn!(event_type, error = %msg, "responses: stream error event");
                round_state.stream_error = Some(msg);
            } else {
                tracing::warn!(
                    event_type,
                    "responses: stream error event without parseable message"
                );
                round_state.stream_error = Some(format!("{event_type} (no details)"));
            }
        }
        other => {
            tracing::debug!(event_type = other, "responses: unhandled SSE event");
        }
    }
}

async fn stream_one_response_round<FO, FR>(
    http: &HttpClient,
    url: &str,
    headers: &[(&str, &str)],
    body: &Value,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
    mut on_delta: FO,
    mut on_reasoning_delta: FR,
) -> Result<(ResponsesStreamRound, String), ResponseError>
where
    FO: FnMut(String) + Send,
    FR: FnMut(String) + Send,
{
    let stream = http
        .post_json_stream_with_headers(url, body, headers, responses_rate_limit_key(options))
        .await
        .map_err(|e| {
            emit_llm_call_error(
                options.status_emitter.as_ref(),
                request_id,
                &options.model,
                round,
            );
            ResponseError::Http(e)
        })?;

    let mut parser = SseParser::new();
    let mut round_state = ResponsesStreamRound::default();
    let model_used = options.model.clone();

    futures_util::pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        if let Some(token) = &options.cancel
            && token.is_cancelled()
        {
            return Err(ResponseError::Cancelled);
        }
        let bytes = chunk.map_err(ResponseError::Http)?;
        let text = String::from_utf8_lossy(&bytes);
        let events = parser.push_str(&text).map_err(ResponseError::Http)?;
        for event in events {
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(data).unwrap_or(Value::Null);
            if apply_chat_completion_chunk_event(&v, &mut round_state, &mut on_delta) {
                continue;
            }
            let event_type = resolve_sse_event_type(event.event.as_deref(), &v);
            apply_responses_stream_event(
                &v,
                &event_type,
                &mut round_state,
                &mut on_delta,
                &mut on_reasoning_delta,
                options.status_emitter.as_ref(),
                request_id,
                round,
                &model_used,
            )
            .await;
        }
    }

    if let Some(err) = round_state.stream_error {
        emit_llm_call_error(
            options.status_emitter.as_ref(),
            request_id,
            &options.model,
            round,
        );
        return Err(ResponseError::StreamFailed(err));
    }

    merge_chat_tool_dispatch(&mut round_state);

    Ok((round_state, model_used))
}

fn emit_llm_call_error(
    status_emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    model: &str,
    round: u32,
) {
    if let Some(emitter) = status_emitter {
        let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallError, request_id, model);
        ev.round = round;
        let emitter = Arc::clone(emitter);
        tokio::spawn(async move {
            emit_safe(Some(&emitter), ev).await;
        });
    }
}

/// Stream a Responses API completion with multi-round tool execution.
///
/// Text deltas are forwarded to `on_delta`; reasoning text to `on_reasoning_delta`.
#[instrument(
    skip(http, registry, hooks, guardrails, caller_messages, options, on_delta, on_reasoning_delta),
    fields(model = %options.model)
)]
pub async fn stream_complete_with_tools<FO, FR>(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    caller_messages: Vec<ChatMessage>,
    options: &ChatOptions,
    mut on_delta: FO,
    mut on_reasoning_delta: FR,
) -> Result<StreamToolOutcome, ResponseError>
where
    FO: FnMut(String) + Send,
    FR: FnMut(String) + Send,
{
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let url = join_base_url(&options.base_url, "/v1/responses");
    let auth = format!("Bearer {}", options.api_key.expose_secret());
    let headers = [("Authorization", auth.as_str())];

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        messages.push(ChatMessage::text("system", sp));
    }
    messages.extend(caller_messages);
    let had_system_prompt = options.system_prompt.is_some();

    if !guardrails.input_is_empty().await {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .map(MessageContent::text_for_summary)
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_default();
        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user
                    && let Some(msg) = messages.iter_mut().rev().find(|m| m.role == "user")
                    && matches!(msg.content, Some(MessageContent::Text(_)))
                {
                    msg.content = Some(MessageContent::Text(transformed));
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(fail_response_partial(
                    ResponseError::Chat(ChatError::Guardrail(GuardrailError::new(
                        GuardrailStage::Input,
                        guard_name,
                        reason,
                    ))),
                    &messages,
                    had_system_prompt,
                ));
            }
        }
    }

    let specs = registry.list_specs().await;
    let tools = if specs.is_empty() {
        None
    } else {
        Some(tools_from_registry(specs))
    };

    let mut api_calls = 0u32;
    let mut previous_response_id: Option<String> = None;
    let mut tool_input: Option<Vec<ResponseInputItem>> = None;
    let loop_guard = new_tool_loop_guard();
    let credentials = credentials_for(options);

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            return Err(fail_response_partial(
                ResponseError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                had_system_prompt,
            ));
        }
        api_calls += 1;
        if let Some(logger) = &options.context_event_logger {
            logger.log(format!(
                "context responses_stream_round={} messages={} chars={}",
                api_calls,
                messages.len(),
                estimate_context_chars(&messages),
            ));
        }

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .map(MessageContent::text_for_summary)
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_default();
        hooks
            .run(
                HookStage::PreCompletion,
                observation_hook_ctx(
                    HookStage::PreCompletion,
                    last_user,
                    &request_id,
                    api_calls,
                    &options.model,
                ),
            )
            .await
            .map_err(|e| {
                fail_response_partial(
                    ResponseError::Chat(ChatError::Hook(e)),
                    &messages,
                    had_system_prompt,
                )
            })?;

        let messages_json_before = serde_json::to_vec(&messages).ok();
        maybe_summarize_messages(
            http,
            &credentials,
            &mut messages,
            &options.summarize_context,
            options.aaak_compression_enabled,
            options.aaak_compression_model.as_deref(),
            &options.model,
            options,
            &request_id,
        )
        .await
        .map_err(|e| fail_response_partial(ResponseError::from(e), &messages, had_system_prompt))?;
        if serde_json::to_vec(&messages).ok() != messages_json_before {
            reset_chain_after_context_mutation(&mut previous_response_id, &mut tool_input);
        }

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        notify_llm_payload(options, api_calls, &request_id, &messages);

        if tool_input.is_some()
            && !previous_response_id
                .as_ref()
                .is_some_and(|id| !id.is_empty())
        {
            tracing::warn!(
                "responses: clearing tool_input; cannot chain without previous_response_id"
            );
            tool_input = None;
        }

        let input = next_response_input(&messages, &previous_response_id, &tool_input);

        let req = ResponseCreateRequest {
            model: options.model.clone(),
            input,
            instructions: options.system_prompt.clone(),
            tools: tools.clone(),
            stream: Some(true),
            previous_response_id: previous_response_id.clone(),
            tool_choice: options.tool_choice.clone(),
            reasoning: reasoning_from_options(options),
        };

        let mut body = serde_json::to_value(&req).map_err(|e| {
            fail_response_partial(ResponseError::Serde(e), &messages, had_system_prompt)
        })?;
        if let Some(extra) = &options.extra_json {
            if let (Value::Object(b), Value::Object(e)) = (&mut body, extra) {
                for (k, v) in e {
                    b.insert(k.clone(), v.clone());
                }
            }
        }

        let (round_state, round_model) = stream_one_response_round(
            http,
            &url,
            &headers,
            &body,
            options,
            &request_id,
            api_calls,
            &mut on_delta,
            &mut on_reasoning_delta,
        )
        .await
        .map_err(|e| fail_response_partial(e, &messages, had_system_prompt))?;
        let model_used = round_model;
        previous_response_id = if round_state.response_id.is_empty() {
            None
        } else {
            Some(round_state.response_id.clone())
        };

        // Prefer completed output when it has real `call_…` ids; streamed SSE can briefly
        // carry only the output item id (`fc_…`) before `call_id` is known.
        let calls = resolve_round_function_calls(&round_state);
        let tool_call_count = calls.len() as u32;
        let estimated_cost = round_state
            .usage
            .as_ref()
            .map(|u| estimate_model_call_cost_usd(&model_used, u));

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
            ev.round = api_calls;
            ev.tool_call_count = tool_call_count;
            ev.usage = round_state.usage.clone();
            ev.estimated_cost_usd = estimated_cost;
            ev
        })
        .await;

        if !calls.is_empty() {
            let tool_calls: Vec<ToolCall> = calls
                .iter()
                .map(|call| ToolCall {
                    id: call.call_id.clone(),
                    kind: "function".to_string(),
                    function: FunctionCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                })
                .collect();

            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: if round_state.content.is_empty() {
                    None
                } else {
                    Some(MessageContent::Text(round_state.content.clone()))
                },
                tool_calls: Some(tool_calls.clone()),
                tool_call_id: None,
                name: None,
                refusal: None,
            });

            let results = futures_util::future::join_all(tool_calls.iter().map(|tc| {
                dispatch_one(
                    tc,
                    hooks,
                    registry,
                    options.status_emitter.as_ref(),
                    &request_id,
                    api_calls,
                    &model_used,
                    false,
                    options.tool_result_max_chars,
                    Some(&loop_guard),
                )
            }))
            .await;

            let mut outputs = Vec::new();
            for (r, tc) in results.into_iter().zip(tool_calls.iter()) {
                let msg = r.map_err(|e| {
                    fail_response_partial(ResponseError::from(e), &messages, had_system_prompt)
                })?;
                messages.push(msg.clone());
                let out_text = msg
                    .content
                    .as_ref()
                    .and_then(|c| c.as_text())
                    .unwrap_or("{}");
                outputs.push(ResponseInputItem::FunctionCallOutput {
                    call_id: tc.id.clone(),
                    output: out_text.to_string(),
                });
            }

            if options.condense_tool_messages {
                let len_before = messages.len();
                condense_tool_round(
                    &mut messages,
                    options.aaak_tool_condensing,
                    options.context_block_provider.as_ref(),
                    options.context_event_logger.as_ref(),
                );
                if messages.len() != len_before {
                    reset_chain_after_context_mutation(&mut previous_response_id, &mut tool_input);
                    continue;
                }
            }
            if previous_response_id
                .as_ref()
                .is_some_and(|id| !id.is_empty())
            {
                tool_input = Some(outputs);
            } else {
                tracing::warn!(
                    "responses: missing response_id after tool round; next call uses full history"
                );
                tool_input = None;
            }
            continue;
        }

        let mut final_content = round_state.content;
        let final_usage = round_state.usage;

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: if final_content.is_empty() {
                None
            } else {
                Some(MessageContent::Text(final_content.clone()))
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
        });

        if !guardrails.output_is_empty().await {
            let (out_outcome, guard_name) = guardrails.run_output(&final_content).await;
            match out_outcome {
                GuardrailOutcome::Allow(transformed) => {
                    final_content = transformed;
                    if let Some(last) = messages.iter_mut().rev().find(|m| m.role == "assistant") {
                        last.content = Some(MessageContent::Text(final_content.clone()));
                    }
                }
                GuardrailOutcome::Block(reason) => {
                    return Err(fail_response_partial(
                        ResponseError::Chat(ChatError::Guardrail(GuardrailError::new(
                            GuardrailStage::Output,
                            guard_name,
                            reason,
                        ))),
                        &messages,
                        had_system_prompt,
                    ));
                }
            }
        }

        break StreamToolOutcome {
            content: final_content,
            finish_reason: Some("stop".into()),
            usage: final_usage,
            request_id: request_id.clone(),
            rounds: api_calls,
            model_used,
            messages: conversation_messages_for_client(&messages, had_system_prompt),
        };
    };

    tracing::info!(
        request_id = %outcome.request_id,
        model = %outcome.model_used,
        rounds = outcome.rounds,
        elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
        "responses stream_complete_with_tools finished"
    );

    Ok(outcome)
}

/// Clear Responses API chaining when local message history was rewritten.
fn reset_chain_after_context_mutation(
    previous_response_id: &mut Option<String>,
    tool_input: &mut Option<Vec<ResponseInputItem>>,
) {
    *previous_response_id = None;
    *tool_input = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::{ImageStore, resolve_message_content};
    use serde_json::json;

    #[test]
    fn reset_chain_clears_previous_response_and_tool_input() {
        let mut prev = Some("resp_123".into());
        let mut tool_input = Some(vec![ResponseInputItem::FunctionCallOutput {
            call_id: "call_1".into(),
            output: "{}".into(),
        }]);
        reset_chain_after_context_mutation(&mut prev, &mut tool_input);
        assert!(prev.is_none());
        assert!(tool_input.is_none());
    }

    fn completed_event_with_message(text: &str) -> Value {
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": text }
                        ]
                    }
                ]
            }
        })
    }

    fn output_item_done_event(text: &str) -> Value {
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    { "type": "output_text", "text": text }
                ]
            }
        })
    }

    fn output_item_done_function_call() -> Value {
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{\"x\":1}"
            }
        })
    }

    fn content_part_done_event(text: &str) -> Value {
        json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": text,
                "annotations": []
            }
        })
    }

    fn output_text_done_event(text: &str) -> Value {
        json!({
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": text
        })
    }

    #[tokio::test]
    async fn completed_event_recovers_message_text_when_no_deltas() {
        let v = completed_event_with_message("# Hello\n\n- **bold** item");
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.completed",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "# Hello\n\n- **bold** item");
        assert_eq!(deltas, vec!["# Hello\n\n- **bold** item".to_string()]);
        assert_eq!(round_state.response_id, "resp_test");
    }

    #[tokio::test]
    async fn completed_event_does_not_override_streamed_content() {
        let v = completed_event_with_message("from completed event");
        let mut round_state = ResponsesStreamRound {
            content: "already streamed".into(),
            ..Default::default()
        };
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.completed",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "already streamed");
        assert!(deltas.is_empty());
    }

    #[tokio::test]
    async fn output_item_done_recovers_message_text() {
        let v = output_item_done_event("hello from item done");
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.output_item.done",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "hello from item done");
        assert_eq!(deltas, vec!["hello from item done".to_string()]);
        assert_eq!(round_state.raw_output.len(), 1);
    }

    #[tokio::test]
    async fn output_item_done_does_not_duplicate_streamed_content() {
        let v = output_item_done_event("from item done");
        let mut round_state = ResponsesStreamRound {
            content: "already streamed".into(),
            ..Default::default()
        };
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.output_item.done",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "already streamed");
        assert!(deltas.is_empty());
    }

    #[tokio::test]
    async fn output_item_done_recovers_function_call() {
        let v = output_item_done_function_call();
        let mut round_state = ResponsesStreamRound::default();

        apply_responses_stream_event(
            &v,
            "response.output_item.done",
            &mut round_state,
            &mut |_delta| {},
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.function_calls.len(), 1);
        assert_eq!(round_state.function_calls[0].name, "echo");
    }

    #[tokio::test]
    async fn completed_event_parses_output_per_item_on_batch_failure() {
        let v = json!({
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            { "type": "output_text", "text": "recovered" }
                        ]
                    },
                    { "type": "unknown_future_type", "data": 1 }
                ]
            }
        });
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.completed",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "recovered");
        assert_eq!(deltas, vec!["recovered".to_string()]);
    }

    #[tokio::test]
    async fn content_part_done_recovers_message_text() {
        let v = content_part_done_event("hello from content part");
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.content_part.done",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "hello from content part");
        assert_eq!(deltas, vec!["hello from content part".to_string()]);
    }

    #[tokio::test]
    async fn output_text_done_recovers_message_text() {
        let v = output_text_done_event("hello from output_text.done");
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();

        apply_responses_stream_event(
            &v,
            "response.output_text.done",
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(round_state.content, "hello from output_text.done");
        assert_eq!(deltas, vec!["hello from output_text.done".to_string()]);
    }

    #[test]
    fn resolve_sse_event_type_prefers_json_then_sse_line() {
        let with_json = json!({"type": "response.completed"});
        assert_eq!(
            resolve_sse_event_type(Some("response.output_item.done"), &with_json),
            "response.completed"
        );
        let without_json = json!({"item": {}});
        assert_eq!(
            resolve_sse_event_type(Some("response.output_item.done"), &without_json),
            "response.output_item.done"
        );
    }

    #[test]
    fn extract_sse_error_message_from_response_failed() {
        let v = json!({
            "type": "response.failed",
            "response": {
                "id": "resp_fail",
                "status": "failed",
                "error": {
                    "code": "invalid_request_error",
                    "message": "context length exceeded"
                }
            }
        });
        assert_eq!(
            extract_sse_error_message(&v),
            Some("invalid_request_error: context length exceeded".to_string())
        );
    }

    #[test]
    fn extract_sse_error_message_from_error_event() {
        let v = json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit reached"
            }
        });
        assert_eq!(
            extract_sse_error_message(&v),
            Some("rate_limit_error: Rate limit reached".to_string())
        );
    }

    #[tokio::test]
    async fn response_failed_sets_stream_error_on_round_state() {
        let v = json!({
            "type": "response.failed",
            "response": {
                "id": "resp_fail",
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "Internal server error"
                }
            }
        });
        let mut round_state = ResponsesStreamRound::default();

        apply_responses_stream_event(
            &v,
            "response.failed",
            &mut round_state,
            &mut |_delta| {},
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;

        assert_eq!(
            round_state.stream_error,
            Some("server_error: Internal server error".to_string())
        );
        assert_eq!(round_state.response_id, "resp_fail");
    }

    #[test]
    fn function_call_output_item_accepts_id_alias() {
        let item = json!({
            "type": "function_call",
            "id": "call_alias",
            "name": "echo",
            "arguments": "{\"x\":1}"
        });
        let parsed: ResponseOutputItem = serde_json::from_value(item).expect("parse");
        match parsed {
            ResponseOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(id.as_deref(), Some("call_alias"));
                assert!(call_id.is_empty());
                assert_eq!(effective_function_call_id(&call_id, &id), "call_alias");
                assert_eq!(name, "echo");
                assert_eq!(arguments, "{\"x\":1}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn chat_completion_chunk_without_id_is_accepted() {
        let chunk = json!({
            "choices": [{
                "index": 0,
                "delta": { "content": "hello" },
                "finish_reason": null
            }]
        });
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();
        assert!(apply_chat_completion_chunk_event(
            &chunk,
            &mut round_state,
            &mut |d| deltas.push(d),
        ));
        assert_eq!(round_state.content, "hello");
        assert_eq!(deltas, vec!["hello".to_string()]);
    }

    #[test]
    fn function_call_with_item_id_and_call_id_deserializes() {
        let j = json!({
            "id": "fc_abc",
            "type": "function_call",
            "status": "completed",
            "arguments": "{\"path\":\".\"}",
            "call_id": "call_xyz",
            "name": "list_dir"
        });
        let parsed = serde_json::from_value::<ResponseOutputItem>(j.clone());
        assert!(parsed.is_ok(), "parse failed: {:?}", parsed.err());
        match parsed.unwrap() {
            ResponseOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(id.as_deref(), Some("fc_abc"));
                assert_eq!(call_id, "call_xyz");
                assert_eq!(name, "list_dir");
                assert_eq!(arguments, "{\"path\":\".\"}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn explicit_call_id_ignores_item_id_alias() {
        let event = json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_abc",
            "name": "list_dir",
            "arguments": "{\"path\":\".\"}"
        });
        assert!(explicit_call_id_from_value(&event).is_none());
        let with_call_id = json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_abc",
            "call_id": "call_xyz",
            "name": "list_dir",
            "arguments": "{\"path\":\".\"}"
        });
        assert_eq!(
            explicit_call_id_from_value(&with_call_id).as_deref(),
            Some("call_xyz")
        );
    }

    #[test]
    fn completed_output_overrides_stale_streamed_function_call_ids() {
        let mut round_state = ResponsesStreamRound::default();
        round_state.function_calls.push(StreamedFunctionCall {
            call_id: "fc_wrong".into(),
            name: "list_dir".into(),
            arguments: "{\"path\":\".\"}".into(),
        });
        round_state.raw_output = vec![ResponseOutputItem::FunctionCall {
            id: Some("fc_abc".into()),
            call_id: "call_xyz".into(),
            name: "list_dir".into(),
            arguments: "{\"path\":\".\"}".into(),
        }];

        let resolved = resolve_round_function_calls(&round_state);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].call_id, "call_xyz");
    }

    #[test]
    fn resolve_prefers_streamed_call_ids_when_completed_only_has_fc() {
        let mut round_state = ResponsesStreamRound::default();
        round_state.function_calls.push(StreamedFunctionCall {
            call_id: "call_xyz".into(),
            name: "list_dir".into(),
            arguments: "{\"path\":\".\"}".into(),
        });
        round_state.raw_output = vec![ResponseOutputItem::FunctionCall {
            id: Some("fc_abc".into()),
            call_id: String::new(),
            name: "list_dir".into(),
            arguments: "{\"path\":\".\"}".into(),
        }];

        let resolved = resolve_round_function_calls(&round_state);
        assert_eq!(resolved[0].call_id, "call_xyz");
    }

    #[test]
    fn resolved_image_parts_map_to_response_input() {
        let mut store = ImageStore::default();
        let hash = store.insert_from_bytes("horse.png", b"\x89PNG\r\n\x1a\n");
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(resolve_message_content(
                &MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "what is this?".into(),
                    },
                    ContentPart::ImageRef {
                        hash,
                        filename: Some("horse.png".into()),
                    },
                ]),
                &store,
            )),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
        }];
        let ResponseInput::Items(items) = chat_messages_to_response_input(&messages) else {
            panic!("expected items");
        };
        let ResponseInputItem::Message { content, .. } = &items[0] else {
            panic!("expected message");
        };
        let arr = content.as_array().expect("multipart content");
        assert!(
            arr.iter()
                .any(|p| p.get("type").and_then(|v| v.as_str()) == Some("input_image"))
        );
    }

    #[test]
    fn chat_messages_to_response_input_maps_user_image_parts() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: crate::openai::ImageUrl {
                        url: "data:image/png;base64,abc".into(),
                        detail: None,
                    },
                },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
        }];
        let ResponseInput::Items(items) = chat_messages_to_response_input(&messages) else {
            panic!("expected items");
        };
        assert_eq!(items.len(), 1);
        let ResponseInputItem::Message { role, content } = &items[0] else {
            panic!("expected message");
        };
        assert_eq!(role, "user");
        let arr = content.as_array().expect("multipart content");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "input_text");
        assert_eq!(arr[1]["type"], "input_image");
        assert_eq!(arr[1]["image_url"], "data:image/png;base64,abc");
    }

    #[test]
    fn chat_messages_to_response_input_drops_orphan_tool_results() {
        let messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_ok".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "echo".into(),
                        arguments: "{}".into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
                refusal: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text("ok".into())),
                tool_calls: None,
                tool_call_id: Some("call_ok".into()),
                name: Some("echo".into()),
                refusal: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text("orphan".into())),
                tool_calls: None,
                tool_call_id: Some("call_missing".into()),
                name: Some("echo".into()),
                refusal: None,
            },
        ];
        let ResponseInput::Items(items) = chat_messages_to_response_input(&messages) else {
            panic!("expected items");
        };
        let outputs: Vec<_> = items
            .iter()
            .filter_map(|i| match i {
                ResponseInputItem::FunctionCallOutput { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs, vec!["call_ok"]);
    }

    #[test]
    fn next_response_input_requires_previous_response_id_to_chain() {
        let messages = vec![ChatMessage::text("user", "hi")];
        let tool_input = Some(vec![ResponseInputItem::FunctionCallOutput {
            call_id: "call_1".into(),
            output: "{}".into(),
        }]);
        let ResponseInput::Items(items) = next_response_input(&messages, &None, &tool_input) else {
            panic!("expected items");
        };
        assert!(
            items
                .iter()
                .all(|i| !matches!(i, ResponseInputItem::FunctionCallOutput { .. })),
            "orphan outputs must not be sent without previous_response_id"
        );

        let ResponseInput::Items(chained) =
            next_response_input(&messages, &Some("resp_1".into()), &tool_input)
        else {
            panic!("expected items");
        };
        assert!(matches!(
            chained.as_slice(),
            [ResponseInputItem::FunctionCallOutput { call_id, .. }] if call_id == "call_1"
        ));
    }
}
