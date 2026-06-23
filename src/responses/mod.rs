//! OpenAI Responses API (`POST /v1/responses`) with tool loop and `previous_response_id` threading.

use std::sync::Arc;
use std::time::Instant;

use futures_util::StreamExt;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::instrument;

use crate::chat::{
    conversation_messages_for_client, credentials_for, post_json_with_model_fallback, ChatError,
    ChatOptions, dispatch_one, observation_hook_ctx, StreamToolOutcome,
};
use crate::costing::estimate_model_call_cost_usd;
use crate::events::{emit_safe, ProcessEvent, ProcessEventKind, StatusEmitter};
use crate::guardrails::{
    GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage,
};
use crate::hooks::{HookRegistry, HookStage};
use crate::http::{join_base_url, HttpClient, sse::SseParser};
use crate::openai::{ChatMessage, FunctionCall, MessageContent, ToolCall, ToolChoice};
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
    OutputText { text: String },
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
            summary: options
                .reasoning_summary
                .api_value()
                .map(str::to_string),
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

fn function_calls(output: &[ResponseOutputItem]) -> Vec<(String, String, String)> {
    output
        .iter()
        .filter_map(|item| {
            if let ResponseOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } = item
            {
                Some((call_id.clone(), name.clone(), arguments.clone()))
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

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            return Err(ResponseError::MaxToolRounds(options.max_tool_rounds));
        }
        api_calls += 1;

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

        emit_safe(
            options.status_emitter.as_ref(),
            {
                let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
                ev.round = api_calls;
                ev
            },
        )
        .await;

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

        let (resp, round_model) = post_response(
            http,
            &url,
            &body,
            &headers,
            options,
            &request_id,
            api_calls,
        )
        .await?;

        model_used = round_model;
        previous_response_id = Some(resp.id.clone());

        let usage_proto = resp.usage.as_ref().map(usage_from_response);
        let estimated_cost = usage_proto
            .as_ref()
            .map(|u| estimate_model_call_cost_usd(&model_used, u));

        let calls = function_calls(&resp.output);
        let tool_call_count = calls.len() as u32;
        emit_safe(
            options.status_emitter.as_ref(),
            {
                let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
                ev.round = api_calls;
                ev.tool_call_count = tool_call_count;
                ev.usage = usage_proto.clone();
                ev.estimated_cost_usd = estimated_cost;
                ev
            },
        )
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

    futures_util::pin_mut!(stream);
    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        let text = String::from_utf8_lossy(&bytes);
        let events = parser
            .push_str(&text)
            .map_err(ResponseError::Http)?;
        for event in events {
            let data = event.data.trim();
            if data.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(data).unwrap_or(Value::Null);
            let event_type = v
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            match event_type {
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
                        });
                    }
                    if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                        response_id = id.to_string();
                    }
                }
                _ => {}
            }
        }
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
#[must_use]
pub(crate) fn chat_messages_to_response_input(messages: &[ChatMessage]) -> ResponseInput {
    let mut items = Vec::new();
    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        if msg.role == "tool" {
            if let Some(call_id) = &msg.tool_call_id {
                let output = msg
                    .content
                    .as_ref()
                    .and_then(|c| c.as_text())
                    .unwrap_or("")
                    .to_string();
                items.push(ResponseInputItem::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output,
                });
            }
            continue;
        }
        if msg.role == "assistant" {
            if let Some(text) = msg
                .content
                .as_ref()
                .and_then(|c| c.as_text())
                .filter(|t| !t.is_empty())
            {
                items.push(ResponseInputItem::Message {
                    role: "assistant".into(),
                    content: Value::String(text.to_string()),
                });
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    if tc.kind == "function" {
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
        if let Some(text) = msg.content.as_ref().and_then(|c| c.as_text()) {
            items.push(ResponseInputItem::Message {
                role: msg.role.clone(),
                content: Value::String(text.to_string()),
            });
        }
    }
    ResponseInput::Items(items)
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
}

fn extract_stream_delta(v: &Value) -> Option<String> {
    v.get("delta")
        .and_then(|d| d.as_str())
        .map(str::to_string)
        .or_else(|| v.pointer("/delta/text").and_then(|t| t.as_str()).map(str::to_string))
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
                call_id,
                name,
                arguments,
            } = item
            {
                Some(StreamedFunctionCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                })
            } else {
                None
            }
        })
        .collect()
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
        "response.function_call_arguments.done" => {
            let call_id = v
                .get("call_id")
                .or_else(|| v.pointer("/item/call_id"))
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
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
                round_state.function_calls.push(call);
                if let Some(emitter) = status_emitter {
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
        "response.completed" => {
            if let Some(u) = v.pointer("/response/usage") {
                round_state.usage = Some(proto::Usage {
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
                });
            }
            if let Some(id) = v.pointer("/response/id").and_then(|id| id.as_str()) {
                round_state.response_id = id.to_string();
            }
            if let Some(output) = v.pointer("/response/output") {
                if let Some(arr) = output.as_array() {
                    if let Ok(parsed) = serde_json::from_value::<Vec<ResponseOutputItem>>(
                        Value::Array(arr.clone()),
                    ) {
                        round_state.raw_output = parsed;
                    }
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
            }
        }
        _ => {}
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
            emit_llm_call_error(options.status_emitter.as_ref(), request_id, &options.model, round);
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
            let event_type = v
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or_default();
            apply_responses_stream_event(
                &v,
                event_type,
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
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user
                    && let Some(msg) = messages.iter_mut().rev().find(|m| m.role == "user")
                {
                    msg.content = Some(MessageContent::Text(transformed));
                }
            }
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
    let mut previous_response_id: Option<String> = None;
    let mut tool_input: Option<Vec<ResponseInputItem>> = None;

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            return Err(ResponseError::MaxToolRounds(options.max_tool_rounds));
        }
        api_calls += 1;

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
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
            .await?;

        emit_safe(
            options.status_emitter.as_ref(),
            {
                let mut ev = ProcessEvent::new(
                    ProcessEventKind::LlmCallStart,
                    &request_id,
                    &options.model,
                );
                ev.round = api_calls;
                ev
            },
        )
        .await;

        let input = if let Some(items) = &tool_input {
            ResponseInput::Items(items.clone())
        } else {
            chat_messages_to_response_input(&messages)
        };

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

        let mut body = serde_json::to_value(&req)?;
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
        .await?;
        let model_used = round_model;
        previous_response_id = Some(round_state.response_id.clone());

        let mut calls = round_state.function_calls;
        if calls.is_empty() {
            calls = function_calls_from_output(&round_state.raw_output);
        }
        let tool_call_count = calls.len() as u32;
        let estimated_cost = round_state
            .usage
            .as_ref()
            .map(|u| estimate_model_call_cost_usd(&model_used, u));

        emit_safe(
            options.status_emitter.as_ref(),
            {
                let mut ev = ProcessEvent::new(
                    ProcessEventKind::LlmCallEnd,
                    &request_id,
                    &model_used,
                );
                ev.round = api_calls;
                ev.tool_call_count = tool_call_count;
                ev.usage = round_state.usage.clone();
                ev.estimated_cost_usd = estimated_cost;
                ev
            },
        )
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
                )
            }))
            .await;

            let mut outputs = Vec::new();
            for (r, tc) in results.into_iter().zip(tool_calls.iter()) {
                let msg = r?;
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
            tool_input = Some(outputs);
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
                    if let Some(last) = messages.iter_mut().rev().find(|m| m.role == "assistant")
                    {
                        last.content = Some(MessageContent::Text(final_content.clone()));
                    }
                }
                GuardrailOutcome::Block(reason) => {
                    return Err(ResponseError::Chat(ChatError::Guardrail(
                        GuardrailError::new(GuardrailStage::Output, guard_name, reason),
                    )));
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
