//! Chat completion proxy: request mapping and upstream forwarding.

use std::sync::Arc;

use futures_util::StreamExt;
use secrecy::SecretString;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::chat::{ChatError, ChatOptions, proxy_chat_post, proxy_chat_stream};
use crate::costing::estimate_model_call_cost_usd;
use crate::gateway::auth::AuthContext;
use crate::gateway::budget::check_budget;
use crate::gateway::db::Database;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::model_access;
use crate::http::{HttpClient, sse::SseParser};
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
};
use crate::proto;
use crate::providers::ProviderId;
use crate::responses::{self, ResponseError};
use crate::tools::ToolSpec;
use serde_json::json;

#[cfg(feature = "capture")]
use crate::gateway::capture::{CaptureApi, CaptureSink, CaptureTap, CaptureTapContext};

/// Bounded buffer for gateway SSE proxy streams (matches gRPC stream channel size).
const GATEWAY_STREAM_BUFFER: usize = 64;

#[cfg(feature = "capture")]
fn capture_max_bytes(capture: &Option<Arc<CaptureSink>>) -> usize {
    capture
        .as_ref()
        .map(|sink| sink.max_response_bytes())
        .unwrap_or(0)
}

fn extract_openai_chat_usage(data: &str) -> Option<proto::Usage> {
    if !data.contains("\"usage\"") {
        return None;
    }
    let chunk: ChatCompletionChunk = serde_json::from_str(data).ok()?;
    chunk.usage.as_ref().map(usage_from_compat)
}

fn extract_openai_responses_usage(data: &str) -> Option<proto::Usage> {
    if !data.contains("\"usage\"") {
        return None;
    }
    let v: Value = serde_json::from_str(data).ok()?;
    v.pointer("/response/usage").map(usage_from_responses_json)
}

async fn record_usage_async(
    db: &Database,
    key_id: Option<&str>,
    user_id: &str,
    model: &str,
    usage: &proto::Usage,
    request_id: &str,
) -> GatewayResult<()> {
    let key_id = key_id.map(str::to_string);
    let user_id = user_id.to_string();
    let model = model.to_string();
    let request_id = request_id.to_string();
    let cost = estimate_model_call_cost_usd(model.as_str(), usage);
    let prompt_tokens = usage.prompt_tokens;
    let completion_tokens = usage.completion_tokens;
    db.run_blocking(move |db| {
        db.record_usage(
            key_id.as_deref(),
            &user_id,
            &model,
            prompt_tokens,
            completion_tokens,
            cost,
            &request_id,
        )
    })
    .await
}

/// Validate model access and budget before proxying (async-safe).
pub async fn preflight_async(
    db: &Database,
    auth: &AuthContext,
    user_id: &str,
    model: &str,
) -> GatewayResult<()> {
    let auth = auth.clone();
    let user_id = user_id.to_string();
    let model = model.to_string();
    db.run_blocking(move |db| preflight(db, &auth, &user_id, &model))
        .await
}

/// Parsed gateway completion request with optional master-key user field.
#[derive(Debug, serde::Deserialize)]
pub struct GatewayCompletionBody {
    #[serde(flatten)]
    pub completion: ChatCompletionRequest,
    /// Required when using the master key; ignored for virtual keys.
    #[serde(default)]
    pub user: Option<String>,
}

/// Parsed gateway Responses request with optional master-key user field.
#[derive(Debug, serde::Deserialize)]
pub struct GatewayResponsesBody {
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, Value>,
    #[serde(default)]
    pub user: Option<String>,
}

fn response_error_to_gateway(err: ResponseError) -> GatewayError {
    chat_error_to_gateway(err.into())
}

fn chat_options_from_responses(body: &GatewayResponsesBody, request_id: &str) -> ChatOptions {
    let mut options = ChatOptions {
        base_url: "https://api.openai.com".into(),
        api_key: SecretString::from(String::new()),
        model: body.model.clone(),
        max_tool_rounds: 0,
        system_prompt: body.instructions.clone(),
        ..ChatOptions::default()
    };
    options.request_id = Some(request_id.to_string());
    let mut extra = body.extra.clone();
    if let Some(reasoning) = extra.remove("reasoning") {
        if let Some(effort) = reasoning.get("effort").and_then(|e| e.as_str()) {
            options.reasoning_effort = Some(effort.to_string());
        }
        if let Some(summary) = reasoning.get("summary").and_then(|s| s.as_str())
            && let Some(level) = crate::chat::reasoning::ReasoningSummaryLevel::parse_str(summary)
        {
            options.reasoning_summary = level;
        }
    }
    if !extra.is_empty() {
        options.extra_json = Some(Value::Object(
            extra
                .into_iter()
                .collect::<serde_json::Map<String, Value>>(),
        ));
    }
    options
}

fn apply_reasoning_cap(options: &mut ChatOptions, auth: &AuthContext) {
    if auth.is_master {
        return;
    }
    let Some(max) = auth.max_reasoning_effort.as_deref() else {
        return;
    };
    if let Some(ref effort) = options.reasoning_effort
        && let Some(clamped) = crate::chat::reasoning::clamp_reasoning_effort_str(effort, max)
    {
        options.reasoning_effort = Some(clamped);
    }
}

fn messages_from_responses_input(input: &Value) -> Vec<ChatMessage> {
    if let Some(text) = input.as_str() {
        return vec![ChatMessage::text("user", text)];
    }
    if let Some(items) = input.as_array() {
        let mut messages = Vec::new();
        for item in items {
            let Some(typ) = item.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            match typ {
                "message" => {
                    let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = item.get("content").map(|c| {
                        if let Some(s) = c.as_str() {
                            s.to_string()
                        } else {
                            c.to_string()
                        }
                    });
                    if let Some(text) = content {
                        messages.push(ChatMessage::text(role, text));
                    }
                }
                _ => {}
            }
        }
        if !messages.is_empty() {
            return messages;
        }
    }
    vec![ChatMessage::text("user", input.to_string())]
}

fn responses_body_value(body: GatewayResponsesBody) -> Value {
    let GatewayResponsesBody {
        model,
        input,
        instructions,
        stream,
        extra,
        user: _,
    } = body;
    let mut map = serde_json::Map::new();
    map.insert("model".into(), json!(model));
    map.insert("input".into(), input);
    if let Some(instructions) = instructions {
        map.insert("instructions".into(), json!(instructions));
    }
    if stream.unwrap_or(false) {
        map.insert("stream".into(), json!(true));
    }
    for (k, v) in extra {
        map.insert(k, v);
    }
    Value::Object(map)
}

fn usage_from_responses_json(u: &Value) -> proto::Usage {
    let input = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let output = u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let total = u
        .get("total_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(u64::from(input) + u64::from(output)) as u32;
    let cached = u
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|x| x.as_u64())
        .map(|n| n as u32)
        .filter(|&n| n > 0);
    crate::usage::usage_from_breakdown(crate::usage::UsageBreakdown {
        prompt_tokens: input,
        completion_tokens: output,
        total_tokens: Some(total),
        cached_tokens: cached,
        reasoning_tokens: None,
    })
}

/// Non-streaming Responses proxy.
pub async fn proxy_response(
    http: &HttpClient,
    credentials: &Arc<crate::providers::ProviderCredentials>,
    db: &Database,
    auth: &AuthContext,
    body: GatewayResponsesBody,
    #[cfg(feature = "capture")] capture: Option<Arc<CaptureSink>>,
) -> GatewayResult<(Value, String)> {
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(auth, body.user.as_deref())?;
    let model = body.model.clone();
    #[cfg(feature = "capture")]
    let max_response_bytes = capture_max_bytes(&capture);
    #[cfg(feature = "capture")]
    let mut tap = CaptureTap::begin(CaptureTapContext {
        sink: capture,
        request_id: request_id.clone(),
        user_id: user_id.clone(),
        key_id: auth.key_id.clone(),
        api: CaptureApi::Responses,
        model_requested: model.clone(),
        stream: false,
        max_response_bytes,
    });
    if let Err(err) = preflight_async(db, auth, &user_id, &model).await {
        #[cfg(feature = "capture")]
        tap.finish_err(err.to_string());
        return Err(err);
    }

    let mut options = chat_options_from_responses(&body, &request_id);
    apply_reasoning_cap(&mut options, auth);
    let messages = messages_from_responses_input(&body.input);
    let req_body = responses_body_value(body);
    #[cfg(feature = "capture")]
    tap.set_request(&req_body);

    let (val, model_ref) = responses::proxy_responses_post(
        http,
        credentials.as_ref(),
        &messages,
        &req_body,
        &options,
        &request_id,
    )
    .await
    .map_err(|e| {
        #[cfg(feature = "capture")]
        tap.finish_err(e.to_string());
        response_error_to_gateway(e)
    })?;

    #[cfg(feature = "capture")]
    {
        tap.set_model_resolved(model_ref.raw.clone());
        tap.set_response(&val);
    }

    if let Some(usage) = val.get("usage") {
        let proto_usage = usage_from_responses_json(usage);
        let cost = estimate_model_call_cost_usd(model_ref.raw.as_str(), &proto_usage);
        record_usage_async(
            db,
            auth.key_id.as_deref(),
            &user_id,
            &model_ref.raw,
            &proto_usage,
            &request_id,
        )
        .await?;
        #[cfg(feature = "capture")]
        tap.finish(Some(&proto_usage), Some(cost));
    } else {
        #[cfg(feature = "capture")]
        tap.finish(None, None);
    }

    Ok((val, request_id))
}

/// Streaming Responses proxy.
pub async fn proxy_response_stream(
    http: &HttpClient,
    credentials: &Arc<crate::providers::ProviderCredentials>,
    db: Database,
    auth: AuthContext,
    body: GatewayResponsesBody,
    #[cfg(feature = "capture")] capture: Option<Arc<CaptureSink>>,
) -> GatewayResult<
    impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static,
> {
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(&auth, body.user.as_deref())?;
    let model = body.model.clone();
    #[cfg(feature = "capture")]
    let max_response_bytes = capture_max_bytes(&capture);
    #[cfg(feature = "capture")]
    let mut tap = CaptureTap::begin(CaptureTapContext {
        sink: capture,
        request_id: request_id.clone(),
        user_id: user_id.clone(),
        key_id: auth.key_id.clone(),
        api: CaptureApi::Responses,
        model_requested: model.clone(),
        stream: true,
        max_response_bytes,
    });
    if let Err(err) = preflight_async(&db, &auth, &user_id, &model).await {
        #[cfg(feature = "capture")]
        tap.finish_err(err.to_string());
        return Err(err);
    }

    let mut options = chat_options_from_responses(&body, &request_id);
    apply_reasoning_cap(&mut options, &auth);
    let messages = messages_from_responses_input(&body.input);
    let req_body = responses_body_value(body);
    #[cfg(feature = "capture")]
    tap.set_request(&req_body);

    let (byte_stream, model_ref) = responses::proxy_responses_stream(
        http,
        credentials.as_ref(),
        &messages,
        &req_body,
        &options,
        &request_id,
    )
    .await
    .map_err(|e| {
        #[cfg(feature = "capture")]
        tap.finish_err(e.to_string());
        response_error_to_gateway(e)
    })?;

    let key_id = auth.key_id.clone();
    let (tx, rx) = mpsc::channel(GATEWAY_STREAM_BUFFER);
    let is_anthropic = model_ref.provider == ProviderId::Anthropic;
    let model_raw = model_ref.raw.clone();

    tokio::spawn(async move {
        let mut stream = byte_stream;
        let mut parser = SseParser::new();
        let mut round_usage: Option<proto::Usage> = None;
        #[cfg(feature = "capture")]
        tap.set_model_resolved(model_raw.clone());

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    if let Ok(events) = parser.push_str(&text) {
                        for event in events {
                            let data = event.data.trim();
                            #[cfg(feature = "capture")]
                            tap.push_sse(data);
                            if data.is_empty() || data == "[DONE]" {
                                continue;
                            }
                            if is_anthropic {
                                let _ = crate::providers::anthropic_stream::AnthropicStreamAccumulator::apply_usage_from_sse_data(
                                    &mut round_usage,
                                    data,
                                );
                            } else if let Some(u) = extract_openai_responses_usage(data) {
                                round_usage = Some(u);
                            }
                        }
                    }
                    if tx.send(Ok(bytes)).await.is_err() {
                        #[cfg(feature = "capture")]
                        tap.finish(None, None);
                        return;
                    }
                }
                Err(e) => {
                    #[cfg(feature = "capture")]
                    tap.finish_err(e.to_string());
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    return;
                }
            }
        }

        if let Some(usage) = round_usage {
            let cost = estimate_model_call_cost_usd(model_raw.as_str(), &usage);
            let _ = record_usage_async(
                &db,
                key_id.as_deref(),
                &user_id,
                &model_raw,
                &usage,
                &request_id,
            )
            .await;
            #[cfg(feature = "capture")]
            tap.finish(Some(&usage), Some(cost));
        } else {
            #[cfg(feature = "capture")]
            tap.finish(None, None);
        }
    });

    Ok(ReceiverStream::new(rx))
}

/// Map an OpenAI-shaped request to [`ChatOptions`] for upstream provider calls.
pub fn chat_options_from_request(req: &ChatCompletionRequest, request_id: &str) -> ChatOptions {
    let mut options = ChatOptions {
        base_url: "https://api.openai.com".into(),
        api_key: SecretString::from(String::new()),
        model: req.model.clone(),
        max_tool_rounds: 0,
        system_prompt: None,
        temperature: req.temperature,
        top_p: req.top_p,
        n: req.n,
        max_completion_tokens: req.max_completion_tokens,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        stop: req.stop.clone(),
        response_format: req.response_format.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
        seed: req.seed,
        store: req.store,
        service_tier: req.service_tier.clone(),
        reasoning_effort: req.reasoning_effort.clone(),
        ..ChatOptions::default()
    };
    options.request_id = Some(request_id.to_string());
    if !req.extra.is_empty() {
        options.extra_json = Some(Value::Object(
            req.extra
                .clone()
                .into_iter()
                .collect::<serde_json::Map<String, Value>>(),
        ));
    }
    options
}

fn tool_specs_from_request(req: &ChatCompletionRequest) -> Option<Vec<ToolSpec>> {
    req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| ToolSpec {
                name: t.function.name.clone(),
                parameters_schema: t.function.parameters.clone(),
                description: t.function.description.clone(),
                static_tool: false,
            })
            .collect()
    })
}

fn usage_from_compat(usage: &crate::openai::Usage) -> proto::Usage {
    crate::usage::usage_from_breakdown(crate::usage::breakdown_from_compat_usage(usage))
}

fn chat_error_to_gateway(err: ChatError) -> GatewayError {
    match err {
        ChatError::Credentials(e) => GatewayError::upstream(e.to_string()),
        ChatError::Http(e) => GatewayError::upstream(e.to_string()),
        ChatError::Api(msg) => GatewayError::upstream(msg),
        ChatError::Serde(e) => GatewayError::upstream(e.to_string()),
        other => GatewayError::upstream(other.to_string()),
    }
}

/// Validate model access and budget before proxying.
pub fn preflight(
    db: &Database,
    auth: &AuthContext,
    user_id: &str,
    model: &str,
) -> GatewayResult<()> {
    if !auth.is_master {
        if !model.contains(':') {
            return Err(GatewayError::forbidden(format!(
                "model must use provider:model format (e.g. openai:gpt-4o-mini); got {model}"
            )));
        }
        let patterns = auth.allowed_models.as_deref().unwrap_or(&[]);
        if !model_access::is_allowed(model, patterns) {
            return Err(GatewayError::forbidden(format!(
                "model {model} is not allowed for this API key"
            )));
        }
    }
    if !db.user_exists(user_id)? {
        return Err(GatewayError::bad_request(format!(
            "user {user_id} does not exist"
        )));
    }
    check_budget(db, user_id)
}

/// Non-streaming completion proxy.
pub async fn proxy_completion(
    http: &HttpClient,
    credentials: &Arc<crate::providers::ProviderCredentials>,
    db: &Database,
    auth: &AuthContext,
    body: GatewayCompletionBody,
    #[cfg(feature = "capture")] capture: Option<Arc<CaptureSink>>,
) -> GatewayResult<(Value, String)> {
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(auth, body.user.as_deref())?;
    #[cfg(feature = "capture")]
    let max_response_bytes = capture_max_bytes(&capture);
    #[cfg(feature = "capture")]
    let mut tap = CaptureTap::begin(CaptureTapContext {
        sink: capture,
        request_id: request_id.clone(),
        user_id: user_id.clone(),
        key_id: auth.key_id.clone(),
        api: CaptureApi::ChatCompletions,
        model_requested: body.completion.model.clone(),
        stream: false,
        max_response_bytes,
    });
    if let Err(err) = preflight_async(db, auth, &user_id, &body.completion.model).await {
        #[cfg(feature = "capture")]
        tap.finish_err(err.to_string());
        return Err(err);
    }

    let mut options = chat_options_from_request(&body.completion, &request_id);
    apply_reasoning_cap(&mut options, auth);
    let tool_specs = tool_specs_from_request(&body.completion);
    let tool_refs = tool_specs.as_deref();
    #[cfg(feature = "capture")]
    if let Ok(request) = serde_json::to_value(&body.completion) {
        tap.set_request(&request);
    }

    let (val, model_ref) = proxy_chat_post(
        http,
        credentials.as_ref(),
        &body.completion.messages,
        tool_refs,
        &options,
        &request_id,
    )
    .await
    .map_err(|e| {
        #[cfg(feature = "capture")]
        tap.finish_err(e.to_string());
        chat_error_to_gateway(e)
    })?;

    let response: ChatCompletionResponse =
        serde_json::from_value(val.clone()).map_err(|e| GatewayError::upstream(e.to_string()))?;

    #[cfg(feature = "capture")]
    {
        tap.set_model_resolved(model_ref.raw.clone());
        tap.set_response(&val);
    }

    if let Some(usage) = &response.usage {
        let proto_usage = usage_from_compat(usage);
        let cost = estimate_model_call_cost_usd(model_ref.raw.as_str(), &proto_usage);
        record_usage_async(
            db,
            auth.key_id.as_deref(),
            &user_id,
            &model_ref.raw,
            &proto_usage,
            &request_id,
        )
        .await?;
        #[cfg(feature = "capture")]
        tap.finish(Some(&proto_usage), Some(cost));
    } else {
        #[cfg(feature = "capture")]
        tap.finish(None, None);
    }

    Ok((val, request_id))
}

/// Streaming completion proxy — forwards SSE bytes and logs usage after the stream ends.
pub async fn proxy_completion_stream(
    http: &HttpClient,
    credentials: &Arc<crate::providers::ProviderCredentials>,
    db: Database,
    auth: AuthContext,
    body: GatewayCompletionBody,
    #[cfg(feature = "capture")] capture: Option<Arc<CaptureSink>>,
) -> GatewayResult<
    impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static,
> {
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(&auth, body.user.as_deref())?;
    let model = body.completion.model.clone();
    #[cfg(feature = "capture")]
    let max_response_bytes = capture_max_bytes(&capture);
    #[cfg(feature = "capture")]
    let mut tap = CaptureTap::begin(CaptureTapContext {
        sink: capture,
        request_id: request_id.clone(),
        user_id: user_id.clone(),
        key_id: auth.key_id.clone(),
        api: CaptureApi::ChatCompletions,
        model_requested: model.clone(),
        stream: true,
        max_response_bytes,
    });
    if let Err(err) = preflight_async(&db, &auth, &user_id, &model).await {
        #[cfg(feature = "capture")]
        tap.finish_err(err.to_string());
        return Err(err);
    }

    let mut options = chat_options_from_request(&body.completion, &request_id);
    apply_reasoning_cap(&mut options, &auth);
    let tool_specs = tool_specs_from_request(&body.completion);
    let tool_refs = tool_specs.as_deref();
    #[cfg(feature = "capture")]
    if let Ok(request) = serde_json::to_value(&body.completion) {
        tap.set_request(&request);
    }

    let (byte_stream, model_ref) = proxy_chat_stream(
        http,
        credentials.as_ref(),
        &body.completion.messages,
        tool_refs,
        &options,
        &request_id,
    )
    .await
    .map_err(|e| {
        #[cfg(feature = "capture")]
        tap.finish_err(e.to_string());
        chat_error_to_gateway(e)
    })?;

    let is_anthropic = model_ref.provider == ProviderId::Anthropic;
    let key_id = auth.key_id.clone();
    let model_raw = model_ref.raw.clone();
    let (tx, rx) = mpsc::channel(GATEWAY_STREAM_BUFFER);

    tokio::spawn(async move {
        let mut stream = byte_stream;
        let mut parser = SseParser::new();
        let mut round_usage: Option<proto::Usage> = None;
        #[cfg(feature = "capture")]
        tap.set_model_resolved(model_raw.clone());

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    if let Ok(events) = parser.push_str(&text) {
                        for event in events {
                            let data = event.data.trim();
                            #[cfg(feature = "capture")]
                            tap.push_sse(data);
                            if data == "[DONE]" {
                                continue;
                            }
                            if is_anthropic {
                                let _ = crate::providers::anthropic_stream::AnthropicStreamAccumulator::apply_usage_from_sse_data(
                                    &mut round_usage,
                                    data,
                                );
                            } else if let Some(u) = extract_openai_chat_usage(data) {
                                round_usage = Some(u);
                            }
                        }
                    }
                    if tx.send(Ok(bytes)).await.is_err() {
                        #[cfg(feature = "capture")]
                        tap.finish(None, None);
                        return;
                    }
                }
                Err(e) => {
                    #[cfg(feature = "capture")]
                    tap.finish_err(e.to_string());
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    return;
                }
            }
        }

        if let Some(usage) = round_usage {
            let cost = estimate_model_call_cost_usd(model_raw.as_str(), &usage);
            let _ = record_usage_async(
                &db,
                key_id.as_deref(),
                &user_id,
                &model_raw,
                &usage,
                &request_id,
            )
            .await;
            #[cfg(feature = "capture")]
            tap.finish(Some(&usage), Some(cost));
        } else {
            #[cfg(feature = "capture")]
            tap.finish(None, None);
        }
    });

    Ok(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::ChatMessage;

    #[test]
    fn maps_chat_options() {
        let req = ChatCompletionRequest::new(
            "openai:gpt-4o-mini".into(),
            vec![ChatMessage::text("user", "hi")],
            None,
        );
        let opts = chat_options_from_request(&req, "req-1");
        assert_eq!(opts.model, "openai:gpt-4o-mini");
        assert_eq!(opts.request_id.as_deref(), Some("req-1"));
    }
}
