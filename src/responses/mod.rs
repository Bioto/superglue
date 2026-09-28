//! OpenAI Responses API (`POST /v1/responses`) with tool loop and `previous_response_id` threading.

use std::sync::Arc;
use std::time::Instant;

use tokio::time::{Duration, sleep};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tracing::instrument;

use crate::chat::{
    ChatError, ChatOptions, DispatchCtx, StreamToolOutcome, condense_tool_round,
    conversation_messages_for_client, credentials_for, dispatch_one, effective_system_prompt,
    estimate_context_chars, fail_partial, injected_system_prefix_count, log_prefix_guard,
    maybe_summarize_messages, messages_with_volatile_suffix, new_tool_loop_guard,
    notify_llm_payload, observation_hook_ctx, prepend_system_messages,
    should_retry_stalled_stream_round, stream_tools,
};
use crate::costing::apply_resolved_cost_usd;
use crate::events::{ProcessEvent, ProcessEventKind, StatusEmitter, emit_safe};
use crate::guardrails::{GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage};
use crate::hooks::{HookRegistry, HookStage};
use crate::http::{HttpClient, sse::SseParser};
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
    injected_prefix: usize,
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
    ResponseError::Chat(fail_partial(chat, messages, injected_prefix))
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
    /// Caller-visible chat history after this turn (for audit/resume).
    pub messages: Vec<ChatMessage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
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

fn response_cache_key(options: &ChatOptions) -> Option<String> {
    options.prompt_cache_key.clone()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseObject {
    #[serde(default)]
    id: String,
    #[serde(default)]
    output: Vec<ResponseOutputItem>,
    usage: Option<ResponseUsage>,
    #[serde(skip)]
    provider_blocks: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseUsageDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
    #[serde(default)]
    input_tokens_details: Option<ResponseUsageDetails>,
    #[serde(default, alias = "cost_usd", alias = "total_cost")]
    cost: Option<f64>,
}

fn tools_from_registry(specs: &[ToolSpec]) -> Vec<ResponseTool> {
    specs
        .iter()
        .map(|s| ResponseTool {
            r#type: "function".to_string(),
            name: s.name.clone(),
            description: s.description.clone(),
            parameters: s.parameters_schema.clone(),
        })
        .collect()
}

#[derive(Serialize)]
struct ResponseCreateRequestRef<'a> {
    model: String,
    input: ResponseInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ResponseTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponseReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
}

fn usage_from_response(u: &ResponseUsage) -> proto::Usage {
    proto::Usage {
        prompt_tokens: u.input_tokens.unwrap_or(0),
        completion_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.total_tokens.unwrap_or(0),
        cached_tokens: u
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .filter(|&n| n > 0),
        reasoning_tokens: None,
        cost_usd: crate::costing::sanitize_billed_cost_usd(u.cost),
    }
}

fn usage_from_response_json(u: &Value) -> proto::Usage {
    let cached = crate::usage::cached_tokens_from_usage_json(u);
    proto::Usage {
        prompt_tokens: u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        completion_tokens: u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        total_tokens: u.get("total_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        cached_tokens: cached,
        reasoning_tokens: None,
        cost_usd: crate::costing::billed_cost_usd_from_usage_json(u),
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

fn initial_previous_response_id(options: &ChatOptions) -> Option<String> {
    options
        .extra_json
        .as_ref()
        .and_then(|extra| extra.get("previous_response_id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn shape_responses_body(body: &mut Value, provider: &dyn crate::providers::LlmProvider) {
    if let Some(obj) = body.as_object_mut() {
        if !provider.supports_previous_response_id() {
            obj.remove("previous_response_id");
            obj.remove("store");
        }
        if !provider.supports_prompt_cache_key() {
            obj.remove("prompt_cache_key");
        }
    }
}

fn response_object_from_normalized(
    normalized: crate::providers::NormalizedResponse,
) -> Result<ResponseObject, ResponseError> {
    let output: Vec<ResponseOutputItem> = normalized
        .output
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default();
    let usage = normalized.usage.map(|u| ResponseUsage {
        input_tokens: Some(u.prompt_tokens),
        output_tokens: Some(u.completion_tokens),
        total_tokens: Some(u.total_tokens),
        input_tokens_details: u.cached_tokens.map(|cached| ResponseUsageDetails {
            cached_tokens: Some(cached),
        }),
        cost: u.cost_usd,
    });
    Ok(ResponseObject {
        id: normalized.id,
        output,
        usage,
        provider_blocks: normalized.provider_blocks,
    })
}

async fn provider_responses_post(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[ToolSpec]>,
    body: &Value,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
) -> Result<(ResponseObject, String), ResponseError> {
    use crate::chat::post_json_cancellable;
    use crate::fallback::{FallbackPolicy, effective_models};

    let models = effective_models(&options.model, options.model_fallback.as_ref());
    let default_policy = FallbackPolicy::default();
    let policy = options
        .model_fallback
        .as_ref()
        .map(|c| &c.policy)
        .unwrap_or(&default_policy);

    let mut last_err = None;
    for (i, model_str) in models.iter().enumerate() {
        if i > 0 {
            crate::fallback::emit_model_fallback(
                options.status_emitter.as_ref(),
                request_id,
                &models[i - 1],
                model_str,
                round,
            )
            .await;
        }

        let model_ref = crate::providers::parse_model_ref(model_str);
        credentials
            .key_for(model_ref.provider)
            .map_err(ChatError::Credentials)?;

        let provider = crate::chat::resolve_chat_provider(&model_ref)?;
        let ctx = crate::providers::ProviderResponsesContext {
            model_ref: &model_ref,
            credentials,
            body,
            messages,
            tools: tool_specs,
            stream: false,
            options,
        };
        let req = provider.build_responses_request(&ctx);
        let mut shaped_body = req.body;
        let base_url = credentials.base_url_for(model_ref.provider);
        let wire_model = crate::providers::wire_model_id(&model_ref, &base_url, model_ref.provider);
        crate::fallback::set_body_model(&mut shaped_body, &wire_model);
        shape_responses_body(&mut shaped_body, provider.as_ref());

        let header_refs: Vec<(&str, &str)> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        match post_json_cancellable(
            http,
            &req.url,
            &shaped_body,
            &header_refs,
            Some(req.rate_limit_key),
            options.cancel.as_ref(),
        )
        .await
        {
            Ok(val) => {
                let normalized = provider.parse_responses_response(&val).map_err(|e| {
                    ResponseError::Chat(ChatError::Http(crate::http::Error::InvalidJson(
                        e.to_string(),
                    )))
                })?;
                let resp = response_object_from_normalized(normalized)?;
                return Ok((resp, model_ref.raw));
            }
            Err(e) => {
                if i + 1 < models.len()
                    && crate::fallback::chat_error_eligible_for_fallback(&e, policy)
                {
                    last_err = Some(e);
                    continue;
                }
                return Err(ResponseError::from(e));
            }
        }
    }
    Err(last_err
        .map(ResponseError::from)
        .unwrap_or(ResponseError::Chat(ChatError::Http(
            crate::http::Error::InvalidJson("model fallback exhausted".into()),
        ))))
}

async fn post_response(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[ToolSpec]>,
    body: &Value,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
) -> Result<(ResponseObject, String), ResponseError> {
    provider_responses_post(
        http,
        credentials,
        messages,
        tool_specs,
        body,
        options,
        request_id,
        round,
    )
    .await
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
    let mut messages: Vec<ChatMessage> = Vec::new();
    let _injected_prefix = prepend_system_messages(options, &mut messages);
    messages.push(ChatMessage::text("user", user_message.clone()));
    responses_tool_loop(
        http,
        registry,
        hooks,
        guardrails,
        messages,
        user_message,
        options,
    )
    .await
}

/// Continue a Responses tool loop from an existing transcript (audit resume).
pub async fn complete_from_messages(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    messages: Vec<ChatMessage>,
    options: &ChatOptions,
) -> Result<ResponseOutcome, ResponseError> {
    let user_message = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text())
        .unwrap_or("")
        .to_string();
    responses_tool_loop(
        http,
        registry,
        hooks,
        guardrails,
        messages,
        user_message,
        options,
    )
    .await
}

async fn responses_tool_loop(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    mut messages: Vec<ChatMessage>,
    user_message: String,
    options: &ChatOptions,
) -> Result<ResponseOutcome, ResponseError> {
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let credentials = credentials_for(options);

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
    let tools_owned = if specs.is_empty() {
        None
    } else {
        Some(tools_from_registry(&specs))
    };

    let mut api_calls = 0u32;
    let mut model_used;
    let mut previous_response_id = initial_previous_response_id(options);
    let mut tool_input: Option<Vec<ResponseInputItem>> = None;
    let loop_guard = new_tool_loop_guard();
    let injected_prefix = injected_system_prefix_count(options);
    let mut last_prefix_hash = None;

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

        let request_messages = messages_with_volatile_suffix(options, &messages);
        log_prefix_guard(
            options,
            if specs.is_empty() {
                None
            } else {
                Some(specs.as_slice())
            },
            &mut last_prefix_hash,
        );
        notify_llm_payload(options, api_calls, &request_id, &request_messages);

        let provider =
            crate::chat::resolve_chat_provider(&crate::providers::parse_model_ref(&options.model))?;
        let input = if provider.supports_previous_response_id() {
            next_response_input(&request_messages, &previous_response_id, &tool_input)
        } else {
            chat_messages_to_response_input(&request_messages)
        };

        let req = ResponseCreateRequestRef {
            model: options.model.clone(),
            input,
            instructions: effective_system_prompt(options),
            tools: tools_owned.as_deref(),
            stream: None,
            previous_response_id: previous_response_id.clone(),
            tool_choice: options.tool_choice.clone(),
            reasoning: reasoning_from_options(options),
            prompt_cache_key: response_cache_key(options),
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
            &credentials,
            &messages,
            if specs.is_empty() {
                None
            } else {
                Some(specs.as_slice())
            },
            &body,
            options,
            &request_id,
            api_calls,
        )
        .await?;

        model_used = round_model;
        previous_response_id = Some(resp.id.clone());

        let mut usage_proto = resp.usage.as_ref().map(usage_from_response);
        let estimated_cost = usage_proto
            .as_mut()
            .map(|u| apply_resolved_cost_usd(&model_used, u, u.cost_usd));

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
            let ctx = DispatchCtx {
                hooks,
                registry,
                status_emitter: options.status_emitter.as_ref(),
                request_id: &request_id,
                round: api_calls,
                model: &model_used,
                emit_start: true,
                tool_result_max_chars: options.tool_result_max_chars,
                loop_guard: Some(&loop_guard),
                cancel: options.cancel.as_ref(),
                code_allowlist: None,
            };
            let results = futures_util::future::join_all(
                tool_calls.iter().map(|tc| dispatch_one(tc, ctx.clone())),
            )
            .await;

            let mut outputs = Vec::new();
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(tool_calls.clone()),
                tool_call_id: None,
                name: None,
                refusal: None,
                provider_blocks: resp.provider_blocks.clone(),
            });
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
                messages.push(msg);
            }
            if let Some(store) = &options.image_store {
                let store = store.lock().await;
                if crate::images::attach_vision_from_tool_results(&mut messages, &store) > 0 {
                    reset_chain_after_context_mutation(&mut previous_response_id, &mut tool_input);
                    continue;
                }
            }
            tool_input = Some(outputs);
            continue;
        }

        let content = extract_output_text(&resp.output);

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.clone().map(MessageContent::Text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
            provider_blocks: resp.provider_blocks.clone(),
        });

        if !guardrails.output_is_empty().await {
            let text = content.clone().unwrap_or_default();
            let (guard_out, guard_name) = guardrails.run_output(&text).await;
            match guard_out {
                GuardrailOutcome::Allow(transformed) => {
                    if let Some(last) = messages.iter_mut().rev().find(|m| m.role == "assistant") {
                        last.content = Some(MessageContent::Text(transformed.clone()));
                    }
                    break ResponseOutcome {
                        id: resp.id,
                        content: Some(transformed),
                        rounds: api_calls,
                        usage: usage_proto,
                        request_id: request_id.clone(),
                        model_used: model_used.clone(),
                        raw_output: resp.output,
                        messages: conversation_messages_for_client(&messages, injected_prefix, 0),
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
            messages: conversation_messages_for_client(&messages, injected_prefix, 0),
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

    let credentials = credentials_for(options);
    let mut messages = vec![ChatMessage::text("user", user_message.clone())];
    let _injected_prefix = prepend_system_messages(options, &mut messages);

    let req = ResponseCreateRequest {
        model: options.model.clone(),
        input: ResponseInput::Text(user_message),
        instructions: effective_system_prompt(options),
        tools: None,
        stream: Some(true),
        previous_response_id: None,
        tool_choice: options.tool_choice.clone(),
        reasoning: reasoning_from_options(options),
        prompt_cache_key: response_cache_key(options),
    };

    let body = serde_json::to_value(&req)?;
    let StreamedRound::Complete(round_state, model_used) = stream_one_response_round(
        http,
        &credentials,
        &messages,
        None,
        &body,
        options,
        &request_id,
        1,
        u32::MAX,
        &mut on_delta,
        &mut |_reasoning: String| {},
    )
    .await?
    else {
        return Err(ResponseError::StreamFailed(
            "stream stalled before first token".into(),
        ));
    };

    let mut content = round_state.content;
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
        id: round_state.response_id,
        content,
        usage: round_state.usage,
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
    provider_blocks: Option<Vec<Value>>,
    /// Set once a terminal SSE event arrives so the reader stops instead of
    /// waiting for the provider to close the connection.
    terminal: bool,
}

enum StreamedRound {
    Complete(Box<ResponsesStreamRound>, String),
    StallBeforeFirstToken,
}

fn responses_round_tools_started(state: &ResponsesStreamRound) -> bool {
    !state.function_calls.is_empty() || !state.chat_tool_dispatch.finish_remaining().is_empty()
}

fn should_retry_responses_stalled_round(
    err: &ChatError,
    stall_retries: u32,
    state: &ResponsesStreamRound,
) -> bool {
    state.reasoning.is_empty()
        && should_retry_stalled_stream_round(
            err,
            stall_retries,
            &state.content,
            responses_round_tools_started(state),
        )
}

fn http_stream_fail(
    err: crate::http::Error,
    stall_retries: u32,
    state: &ResponsesStreamRound,
) -> Result<StreamedRound, ResponseError> {
    let chat_err = ChatError::Http(err);
    if should_retry_responses_stalled_round(&chat_err, stall_retries, state) {
        Ok(StreamedRound::StallBeforeFirstToken)
    } else if let ChatError::Http(http_err) = chat_err {
        Err(ResponseError::Http(http_err))
    } else {
        Err(ResponseError::Chat(chat_err))
    }
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

fn ingest_stream_output_item(
    parsed: ResponseOutputItem,
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
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
                call_id: resolved_call_id,
                name,
                arguments,
            };
            // Do not emit ToolCallStart here — the stream may still be open for
            // seconds after args complete. dispatch_one emits start when the tool
            // actually runs so the UI does not show a stuck "running" row.
            let _ = push_streamed_function_call(round_state, call);
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
    chunk: &ChatCompletionChunk,
    round_state: &mut ResponsesStreamRound,
    on_delta: &mut impl FnMut(String),
    on_reasoning_delta: &mut impl FnMut(String),
) -> bool {
    if chunk.choices.is_empty() && chunk.usage.is_none() {
        return false;
    }
    if !chunk.id.is_empty() {
        round_state.response_id = chunk.id.clone();
    }
    for choice in &chunk.choices {
        if let Some(text) = choice.delta.reasoning_text() {
            round_state.reasoning.push_str(&text);
            on_reasoning_delta(text);
        }
    }
    let prev_len = round_state.content.len();
    let _ready = stream_tools::apply_openai_chunk(
        chunk,
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
                        name,
                        arguments,
                    };
                    // Record the call only; ToolCallStart waits for dispatch_one.
                    let _ = push_streamed_function_call(round_state, call);
                }
            }
        }
        "response.output_item.done" => {
            if let Some(item) = v.get("item") {
                if let Ok(parsed) = serde_json::from_value::<ResponseOutputItem>(item.clone()) {
                    ingest_stream_output_item(parsed, round_state, on_delta);
                } else {
                    tracing::debug!("responses: failed to parse output_item.done item");
                }
            }
        }
        "response.completed" | "response.incomplete" => {
            round_state.terminal = true;
            if let Some(u) = v.pointer("/response/usage") {
                round_state.usage = Some(usage_from_response_json(u));
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
            round_state.terminal = true;
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
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[ToolSpec]>,
    body: &Value,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
    stall_retries: u32,
    mut on_delta: FO,
    mut on_reasoning_delta: FR,
) -> Result<StreamedRound, ResponseError>
where
    FO: FnMut(String) + Send,
    FR: FnMut(String) + Send,
{
    use crate::fallback::{FallbackPolicy, effective_models};

    let models = effective_models(&options.model, options.model_fallback.as_ref());
    let default_policy = FallbackPolicy::default();
    let policy = options
        .model_fallback
        .as_ref()
        .map(|c| &c.policy)
        .unwrap_or(&default_policy);

    let mut last_err = None;
    for (i, model_str) in models.iter().enumerate() {
        if i > 0 {
            crate::fallback::emit_model_fallback(
                options.status_emitter.as_ref(),
                request_id,
                &models[i - 1],
                model_str,
                round,
            )
            .await;
        }

        let model_ref = crate::providers::parse_model_ref(model_str);
        if let Err(e) = credentials.key_for(model_ref.provider) {
            return Err(ResponseError::Chat(ChatError::Credentials(e)));
        }

        let provider = crate::chat::resolve_chat_provider(&model_ref)?;
        let ctx = crate::providers::ProviderResponsesContext {
            model_ref: &model_ref,
            credentials,
            body,
            messages,
            tools: tool_specs,
            stream: true,
            options,
        };
        let req = provider.build_responses_request(&ctx);
        let mut shaped_body = req.body;
        let base_url = credentials.base_url_for(model_ref.provider);
        let wire_model = crate::providers::wire_model_id(&model_ref, &base_url, model_ref.provider);
        crate::fallback::set_body_model(&mut shaped_body, &wire_model);
        shape_responses_body(&mut shaped_body, provider.as_ref());
        let header_refs: Vec<(&str, &str)> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let stream = match http
            .post_json_stream_with_headers(
                &req.url,
                &shaped_body,
                &header_refs,
                Some(req.rate_limit_key),
            )
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                let chat_err = ChatError::Http(e);
                if i + 1 < models.len()
                    && crate::fallback::chat_error_eligible_for_fallback(&chat_err, policy)
                {
                    emit_llm_call_error(
                        options.status_emitter.as_ref(),
                        request_id,
                        model_str,
                        round,
                    );
                    last_err = Some(ResponseError::Chat(chat_err));
                    continue;
                }
                emit_llm_call_error(
                    options.status_emitter.as_ref(),
                    request_id,
                    model_str,
                    round,
                );
                if should_retry_responses_stalled_round(
                    &chat_err,
                    stall_retries,
                    &ResponsesStreamRound::default(),
                ) {
                    return Ok(StreamedRound::StallBeforeFirstToken);
                }
                return Err(ResponseError::Chat(chat_err));
            }
        };

        let mut round_state = ResponsesStreamRound::default();
        let model_used = model_ref.raw.clone();

        if model_ref.provider == crate::providers::ProviderId::Anthropic {
            let mut acc = crate::providers::anthropic_stream::AnthropicStreamAccumulator::new();
            let mut parser = SseParser::new();
            futures_util::pin_mut!(stream);
            while let Some(chunk) = stream.next().await {
                if let Some(token) = &options.cancel
                    && token.is_cancelled()
                {
                    return Err(ResponseError::Cancelled);
                }
                let bytes = match chunk {
                    Ok(bytes) => bytes,
                    Err(e) => return http_stream_fail(e, stall_retries, &round_state),
                };
                let text = String::from_utf8_lossy(&bytes);
                let events = match parser.push_str(&text) {
                    Ok(events) => events,
                    Err(e) => return http_stream_fail(e, stall_retries, &round_state),
                };
                for event in events {
                    let data = event.data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    if let Ok(Some(delta)) = acc.apply_sse_data(data) {
                        match delta {
                            crate::providers::anthropic_stream::AnthropicStreamDelta::Text(t) => {
                                on_delta(t);
                            }
                            crate::providers::anthropic_stream::AnthropicStreamDelta::Thinking(
                                t,
                            ) => {
                                round_state.reasoning.push_str(&t);
                                on_reasoning_delta(t);
                            }
                        }
                    }
                    if crate::providers::anthropic_stream::AnthropicStreamAccumulator::
                        is_terminal_sse_data(data)
                    {
                        round_state.terminal = true;
                    }
                }
                if round_state.terminal {
                    break;
                }
            }
            let thinking = acc.thinking_text().to_string();
            let outcome = acc.into_round_outcome();
            round_state.response_id = uuid::Uuid::new_v4().to_string();
            round_state.content = outcome.content;
            round_state.reasoning = thinking;
            round_state.usage = outcome.usage;
            round_state.provider_blocks = outcome.provider_blocks.clone();
            for tc in outcome.tool_calls {
                round_state.function_calls.push(StreamedFunctionCall {
                    call_id: tc.id,
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                });
            }
            return Ok(StreamedRound::Complete(Box::new(round_state), model_used));
        }

        let mut parser = SseParser::new();

        futures_util::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            if let Some(token) = &options.cancel
                && token.is_cancelled()
            {
                return Err(ResponseError::Cancelled);
            }
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(e) => return http_stream_fail(e, stall_retries, &round_state),
            };
            let text = String::from_utf8_lossy(&bytes);
            let events = match parser.push_str(&text) {
                Ok(events) => events,
                Err(e) => return http_stream_fail(e, stall_retries, &round_state),
            };
            for event in events {
                let data = event.data.trim();
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    round_state.terminal = true;
                    break;
                }
                if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(data) {
                    if apply_chat_completion_chunk_event(
                        &chunk,
                        &mut round_state,
                        &mut on_delta,
                        &mut on_reasoning_delta,
                    ) {
                        continue;
                    }
                }
                let v: Value = serde_json::from_str(data).unwrap_or(Value::Null);
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
            // The response is finished; do not wait for the provider to close the
            // body (that idle tail added seconds before the turn could complete).
            if round_state.terminal {
                break;
            }
        }

        if let Some(err) = round_state.stream_error.as_ref() {
            emit_llm_call_error(
                options.status_emitter.as_ref(),
                request_id,
                model_str,
                round,
            );
            let chat_err = ChatError::Api(err.clone());
            if should_retry_responses_stalled_round(&chat_err, stall_retries, &round_state) {
                return Ok(StreamedRound::StallBeforeFirstToken);
            }
            return Err(ResponseError::StreamFailed(err.clone()));
        }

        merge_chat_tool_dispatch(&mut round_state);

        return Ok(StreamedRound::Complete(Box::new(round_state), model_used));
    }

    let err = last_err.unwrap_or(ResponseError::Chat(ChatError::Http(
        crate::http::Error::InvalidJson("model fallback exhausted".into()),
    )));
    if let ResponseError::Chat(ref chat_err) = err
        && should_retry_responses_stalled_round(
            chat_err,
            stall_retries,
            &ResponsesStreamRound::default(),
        )
    {
        return Ok(StreamedRound::StallBeforeFirstToken);
    }
    Err(err)
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

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    let injected_prefix = prepend_system_messages(options, &mut messages);
    messages.extend(caller_messages);

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
                    injected_prefix,
                ));
            }
        }
    }

    let specs = registry.list_specs().await;
    let tools_owned = if specs.is_empty() {
        None
    } else {
        Some(tools_from_registry(&specs))
    };

    let mut api_calls = 0u32;
    let mut last_prompt_tokens: Option<u32> = None;
    let mut previous_response_id = initial_previous_response_id(options);
    let mut tool_input: Option<Vec<ResponseInputItem>> = None;
    let mut stall_retries = 0u32;
    let loop_guard = new_tool_loop_guard();
    let mut last_prefix_hash = None;
    let credentials = credentials_for(options);

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            return Err(fail_response_partial(
                ResponseError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                injected_prefix,
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
                    injected_prefix,
                )
            })?;

        let mutated = maybe_summarize_messages(
            http,
            &credentials,
            &mut messages,
            &options.summarize_context,
            options.aaak_compression_enabled,
            options.aaak_compression_model.as_deref(),
            &options.model,
            options,
            &request_id,
            last_prompt_tokens,
        )
        .await
        .map_err(|e| fail_response_partial(ResponseError::from(e), &messages, injected_prefix))?;
        if mutated {
            reset_chain_after_context_mutation(&mut previous_response_id, &mut tool_input);
        }

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        let request_messages = messages_with_volatile_suffix(options, &messages);
        log_prefix_guard(
            options,
            if specs.is_empty() {
                None
            } else {
                Some(specs.as_slice())
            },
            &mut last_prefix_hash,
        );
        notify_llm_payload(options, api_calls, &request_id, &request_messages);

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

        let provider =
            crate::chat::resolve_chat_provider(&crate::providers::parse_model_ref(&options.model))?;
        let input = if provider.supports_previous_response_id() {
            next_response_input(&request_messages, &previous_response_id, &tool_input)
        } else {
            chat_messages_to_response_input(&request_messages)
        };

        let req = ResponseCreateRequestRef {
            model: options.model.clone(),
            input,
            instructions: effective_system_prompt(options),
            tools: tools_owned.as_deref(),
            stream: Some(true),
            previous_response_id: previous_response_id.clone(),
            tool_choice: options.tool_choice.clone(),
            reasoning: reasoning_from_options(options),
            prompt_cache_key: response_cache_key(options),
        };

        let mut body = serde_json::to_value(&req).map_err(|e| {
            fail_response_partial(ResponseError::Serde(e), &messages, injected_prefix)
        })?;
        if let Some(extra) = &options.extra_json {
            if let (Value::Object(b), Value::Object(e)) = (&mut body, extra) {
                for (k, v) in e {
                    b.insert(k.clone(), v.clone());
                }
            }
        }

        let (mut round_state, round_model) = match stream_one_response_round(
            http,
            &credentials,
            &request_messages,
            if specs.is_empty() {
                None
            } else {
                Some(specs.as_slice())
            },
            &body,
            options,
            &request_id,
            api_calls,
            stall_retries,
            &mut on_delta,
            &mut on_reasoning_delta,
        )
        .await
        {
            Ok(StreamedRound::Complete(state, model)) => (*state, model),
            Ok(StreamedRound::StallBeforeFirstToken) => {
                stall_retries += 1;
                api_calls = api_calls.saturating_sub(1);
                tracing::warn!(
                    request_id = %request_id,
                    model = %options.model,
                    attempt = stall_retries,
                    "LLM stream stalled before first token — retrying round"
                );
                sleep(Duration::from_millis(400)).await;
                continue;
            }
            Err(e) => {
                return Err(fail_response_partial(e, &messages, injected_prefix));
            }
        };
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
            .as_mut()
            .map(|u| apply_resolved_cost_usd(&model_used, u, u.cost_usd));

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
            ev.round = api_calls;
            ev.tool_call_count = tool_call_count;
            ev.usage = round_state.usage.clone();
            ev.estimated_cost_usd = estimated_cost;
            ev
        })
        .await;
        if let Some(tokens) = round_state
            .usage
            .as_ref()
            .map(|usage| usage.prompt_tokens)
            .filter(|tokens| *tokens > 0)
        {
            last_prompt_tokens = Some(tokens);
        }

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
                provider_blocks: round_state.provider_blocks.clone(),
            });

            let ctx = DispatchCtx {
                hooks,
                registry,
                status_emitter: options.status_emitter.as_ref(),
                request_id: &request_id,
                round: api_calls,
                model: &model_used,
                emit_start: true,
                tool_result_max_chars: options.tool_result_max_chars,
                loop_guard: Some(&loop_guard),
                cancel: options.cancel.as_ref(),
                code_allowlist: None,
            };
            let results = futures_util::future::join_all(
                tool_calls.iter().map(|tc| dispatch_one(tc, ctx.clone())),
            )
            .await;

            let mut outputs = Vec::new();
            for (r, tc) in results.into_iter().zip(tool_calls.iter()) {
                let msg = r.map_err(|e| {
                    fail_response_partial(ResponseError::from(e), &messages, injected_prefix)
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

            if let Some(store) = &options.image_store {
                let store = store.lock().await;
                if crate::images::attach_vision_from_tool_results(&mut messages, &store) > 0 {
                    reset_chain_after_context_mutation(&mut previous_response_id, &mut tool_input);
                    continue;
                }
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
            provider_blocks: round_state.provider_blocks.clone(),
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
                        injected_prefix,
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
            messages: conversation_messages_for_client(&messages, injected_prefix, 0),
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

/// Gateway-facing single-hop Responses proxy (non-streaming).
#[cfg(feature = "gateway")]
pub async fn proxy_responses_post(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    body: &Value,
    options: &ChatOptions,
    request_id: &str,
) -> Result<(Value, crate::providers::ModelRef), ResponseError> {
    let (resp, model_used) = provider_responses_post(
        http,
        credentials,
        messages,
        None,
        body,
        options,
        request_id,
        1,
    )
    .await?;
    let val = serde_json::to_value(&resp).map_err(ResponseError::Serde)?;
    let model_ref = crate::providers::parse_model_ref(&model_used);
    Ok((val, model_ref))
}

/// Gateway-facing single-hop Responses proxy (streaming).
#[cfg(feature = "gateway")]
pub async fn proxy_responses_stream(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    body: &Value,
    options: &ChatOptions,
    _request_id: &str,
) -> Result<
    (
        impl futures_util::Stream<Item = Result<bytes::Bytes, crate::http::Error>> + Send + use<>,
        crate::providers::ModelRef,
    ),
    ResponseError,
> {
    let model_ref = crate::providers::parse_model_ref(&options.model);
    credentials
        .key_for(model_ref.provider)
        .map_err(|e| ResponseError::Chat(ChatError::Credentials(e)))?;

    let provider = crate::chat::resolve_chat_provider(&model_ref)?;
    let ctx = crate::providers::ProviderResponsesContext {
        model_ref: &model_ref,
        credentials,
        body,
        messages,
        tools: None,
        stream: true,
        options,
    };
    let req = provider.build_responses_request(&ctx);
    let mut shaped_body = req.body;
    shape_responses_body(&mut shaped_body, provider.as_ref());
    let header_refs: Vec<(&str, &str)> = req
        .headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let stream = http
        .post_json_stream_with_headers(
            &req.url,
            &shaped_body,
            &header_refs,
            Some(req.rate_limit_key),
        )
        .await
        .map_err(ResponseError::Http)?;
    Ok((stream, model_ref))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::images::{ImageStore, resolve_message_content};
    use crate::providers::resolve_provider;
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
    async fn terminal_events_stop_the_stream_reader() {
        for event_type in [
            "response.completed",
            "response.incomplete",
            "response.failed",
        ] {
            let mut round_state = ResponsesStreamRound::default();
            apply_responses_stream_event(
                &json!({ "type": event_type, "response": { "id": "resp_test" } }),
                event_type,
                &mut round_state,
                &mut |_delta| {},
                &mut |_reasoning| {},
                None,
                "req-1",
                1,
                "gpt-test",
            )
            .await;
            assert!(round_state.terminal, "{event_type} should end the round");
        }

        let mut mid_round = ResponsesStreamRound::default();
        apply_responses_stream_event(
            &json!({ "type": "response.output_text.delta", "delta": "hi" }),
            "response.output_text.delta",
            &mut mid_round,
            &mut |_delta| {},
            &mut |_reasoning| {},
            None,
            "req-1",
            1,
            "gpt-test",
        )
        .await;
        assert!(!mid_round.terminal, "deltas must not end the round");
    }

    #[test]
    fn anthropic_message_stop_is_terminal() {
        use crate::providers::anthropic_stream::AnthropicStreamAccumulator as Acc;
        assert!(Acc::is_terminal_sse_data(r#"{"type":"message_stop"}"#));
        assert!(Acc::is_terminal_sse_data("[DONE]"));
        assert!(!Acc::is_terminal_sse_data(
            r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}"#
        ));
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
    fn responses_stall_retry_before_first_token() {
        let err = ChatError::Api("error decoding response body".into());
        let empty = ResponsesStreamRound::default();
        assert!(should_retry_responses_stalled_round(&err, 0, &empty));
        assert!(should_retry_responses_stalled_round(&err, 1, &empty));
    }

    #[test]
    fn responses_stall_retry_respects_max_attempts() {
        let err = ChatError::Api("error decoding response body".into());
        let empty = ResponsesStreamRound::default();
        assert!(!should_retry_responses_stalled_round(&err, 2, &empty));
    }

    #[test]
    fn responses_stall_retry_skips_after_tokens_or_tools() {
        let err = ChatError::Api("error decoding response body".into());
        let with_content = ResponsesStreamRound {
            content: "hello".into(),
            ..ResponsesStreamRound::default()
        };
        assert!(!should_retry_responses_stalled_round(
            &err,
            0,
            &with_content
        ));

        let with_reasoning = ResponsesStreamRound {
            reasoning: "thinking".into(),
            ..ResponsesStreamRound::default()
        };
        assert!(!should_retry_responses_stalled_round(
            &err,
            0,
            &with_reasoning
        ));

        let with_tools = ResponsesStreamRound {
            function_calls: vec![StreamedFunctionCall {
                call_id: "call_1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }],
            ..ResponsesStreamRound::default()
        };
        assert!(!should_retry_responses_stalled_round(&err, 0, &with_tools));
        assert!(responses_round_tools_started(&with_tools));
        assert!(!responses_round_tools_started(
            &ResponsesStreamRound::default()
        ));
    }

    #[test]
    fn responses_stall_retry_skips_non_transient_stream_error() {
        let err = ChatError::Api("server_error: Internal server error".into());
        let empty = ResponsesStreamRound::default();
        assert!(!should_retry_responses_stalled_round(&err, 0, &empty));

        let gateway = ChatError::Http(crate::http::Error::Unsuccessful {
            status: reqwest::StatusCode::BAD_GATEWAY,
            preview: "error code: 502".into(),
            len: 16,
            retry_after: None,
        });
        assert!(!should_retry_responses_stalled_round(&gateway, 0, &empty));
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
        let chunk: ChatCompletionChunk = serde_json::from_value(json!({
            "choices": [{
                "index": 0,
                "delta": { "content": "hello" },
                "finish_reason": null
            }]
        }))
        .unwrap();
        let mut round_state = ResponsesStreamRound::default();
        let mut deltas = Vec::new();
        assert!(apply_chat_completion_chunk_event(
            &chunk,
            &mut round_state,
            &mut |d| deltas.push(d),
            &mut |_| {},
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
            provider_blocks: None,
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
            provider_blocks: None,
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
                provider_blocks: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text("ok".into())),
                tool_calls: None,
                tool_call_id: Some("call_ok".into()),
                name: Some("echo".into()),
                refusal: None,
                provider_blocks: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text("orphan".into())),
                tool_calls: None,
                tool_call_id: Some("call_missing".into()),
                name: Some("echo".into()),
                refusal: None,
                provider_blocks: None,
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

    #[test]
    fn response_body_includes_previous_response_id_from_extra_json() {
        let options = ChatOptions {
            extra_json: Some(json!({"previous_response_id": "resp_tool"})),
            model: "mock".into(),
            ..Default::default()
        };
        let prev = initial_previous_response_id(&options);
        assert_eq!(prev.as_deref(), Some("resp_tool"));

        let req = ResponseCreateRequest {
            model: options.model.clone(),
            input: ResponseInput::Text("continue".into()),
            instructions: None,
            tools: None,
            stream: None,
            previous_response_id: prev,
            tool_choice: None,
            reasoning: None,
            prompt_cache_key: None,
        };
        let mut body = serde_json::to_value(&req).unwrap();
        if let Some(extra) = &options.extra_json {
            if let (Value::Object(b), Value::Object(e)) = (&mut body, extra) {
                for (k, v) in e {
                    b.insert(k.clone(), v.clone());
                }
            }
        }
        let provider =
            resolve_provider(&crate::providers::parse_model_ref("mock")).expect("openai compat");
        shape_responses_body(&mut body, provider.as_ref());
        assert_eq!(
            body.get("previous_response_id").and_then(|v| v.as_str()),
            Some("resp_tool")
        );
    }

    #[test]
    fn usage_from_response_json_reads_cached_tokens() {
        let usage = usage_from_response_json(&json!({
            "input_tokens": 120,
            "output_tokens": 10,
            "total_tokens": 130,
            "input_tokens_details": { "cached_tokens": 90 }
        }));
        assert_eq!(usage.prompt_tokens, 120);
        assert_eq!(usage.cached_tokens, Some(90));
    }

    #[test]
    fn shape_responses_body_strips_prev_id_for_groq() {
        let mut body = json!({
            "model": "groq:llama",
            "previous_response_id": "resp_1",
            "store": true,
            "prompt_cache_key": "key"
        });
        let provider = resolve_provider(&crate::providers::parse_model_ref("groq:llama"))
            .expect("groq is chat");
        shape_responses_body(&mut body, provider.as_ref());
        assert!(body.get("previous_response_id").is_none());
        assert!(body.get("store").is_none());
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn shape_responses_body_strips_prompt_cache_key_for_anthropic() {
        let mut body = json!({
            "model": "anthropic:claude",
            "prompt_cache_key": "key"
        });
        let provider = resolve_provider(&crate::providers::parse_model_ref(
            "anthropic:claude-sonnet-4",
        ))
        .expect("anthropic is chat");
        shape_responses_body(&mut body, provider.as_ref());
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn shape_responses_body_keeps_fields_for_openai_and_xai() {
        let mut openai_body = json!({
            "model": "gpt-4o",
            "previous_response_id": "resp_1",
            "store": true,
            "prompt_cache_key": "key"
        });
        let openai = resolve_provider(&crate::providers::parse_model_ref("openai:gpt-4o"))
            .expect("openai is chat");
        shape_responses_body(&mut openai_body, openai.as_ref());
        assert_eq!(
            openai_body
                .get("previous_response_id")
                .and_then(|v| v.as_str()),
            Some("resp_1")
        );
        assert_eq!(
            openai_body.get("store").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            openai_body.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some("key")
        );

        let mut xai_body = json!({
            "model": "xai:grok",
            "previous_response_id": "resp_2",
            "prompt_cache_key": "conv"
        });
        let xai = resolve_provider(&crate::providers::parse_model_ref("xai:grok-4"))
            .expect("xai is chat");
        shape_responses_body(&mut xai_body, xai.as_ref());
        assert_eq!(
            xai_body
                .get("previous_response_id")
                .and_then(|v| v.as_str()),
            Some("resp_2")
        );
        assert_eq!(
            xai_body.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some("conv")
        );
    }
}
