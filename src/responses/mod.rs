//! OpenAI Responses API (`POST /v1/responses`) with tool loop and `previous_response_id` threading.

use std::time::Instant;

use futures_util::StreamExt;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::instrument;

use crate::chat::{
    credentials_for, post_json_with_model_fallback, ChatError, ChatOptions, dispatch_one,
    observation_hook_ctx,
};
use crate::costing::estimate_model_call_cost_usd;
use crate::events::{emit_safe, ProcessEvent, ProcessEventKind};
use crate::guardrails::{
    GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage,
};
use crate::hooks::{HookRegistry, HookStage};
use crate::http::{join_base_url, HttpClient, sse::SseParser};
use crate::openai::{FunctionCall, ToolCall, ToolChoice};
use crate::proto;
use crate::tools::{ToolRegistry, ToolSpec};

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
    reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ResponseInput {
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
            ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model),
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
            reasoning_effort: options.reasoning_effort.clone(),
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

        emit_safe(
            options.status_emitter.as_ref(),
            {
                let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
                ev.round = api_calls;
                ev.usage = usage_proto.clone();
                ev.estimated_cost_usd = estimated_cost;
                ev
            },
        )
        .await;

        let calls = function_calls(&resp.output);
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
        reasoning_effort: options.reasoning_effort.clone(),
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
