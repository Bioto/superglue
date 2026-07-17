//! Chat completion proxy: request mapping and upstream forwarding.

use std::sync::Arc;

use futures_util::StreamExt;
use secrecy::SecretString;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::chat::{proxy_chat_post, proxy_chat_stream, ChatError, ChatOptions};
use crate::costing::estimate_model_call_cost_usd;
use crate::gateway::auth::AuthContext;
use crate::gateway::budget::check_budget;
use crate::gateway::db::Database;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::model_access::{self, is_unrestricted};
use crate::http::{HttpClient, sse::SseParser};
use crate::openai::{ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk};
use crate::proto;
use crate::providers::ProviderId;
use crate::tools::ToolSpec;

/// Parsed gateway completion request with optional master-key user field.
#[derive(Debug, serde::Deserialize)]
pub struct GatewayCompletionBody {
    #[serde(flatten)]
    pub completion: ChatCompletionRequest,
    /// Required when using the master key; ignored for virtual keys.
    #[serde(default)]
    pub user: Option<String>,
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
        options.extra_json = Some(serde_json::to_value(&req.extra).unwrap_or(Value::Null));
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

fn usage_from_openai(usage: &crate::openai::Usage) -> proto::Usage {
    proto::Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
    }
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
        let patterns = auth.allowed_models.as_deref().unwrap_or(&[]);
        if !model_access::is_allowed(model, patterns) {
            return Err(GatewayError::forbidden(format!(
                "model {model} is not allowed for this API key"
            )));
        }
    }
    if !db.user_exists(user_id)? {
        return Err(GatewayError::bad_request(format!("user {user_id} does not exist")));
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
) -> GatewayResult<(Value, String)> {
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(auth, body.user.as_deref())?;
    preflight(db, auth, &user_id, &body.completion.model)?;

    let options = chat_options_from_request(&body.completion, &request_id);
    let tool_specs = tool_specs_from_request(&body.completion);
    let tool_refs = tool_specs.as_deref();

    let (val, model_ref) = proxy_chat_post(
        http,
        credentials.as_ref(),
        &body.completion.messages,
        tool_refs,
        &options,
        &request_id,
    )
    .await
    .map_err(chat_error_to_gateway)?;

    let response: ChatCompletionResponse =
        serde_json::from_value(val.clone()).map_err(|e| GatewayError::upstream(e.to_string()))?;

    if let Some(usage) = &response.usage {
        let proto_usage = usage_from_openai(usage);
        let cost = estimate_model_call_cost_usd(&body.completion.model, &proto_usage);
        db.record_usage(
            auth.key_id.as_deref(),
            &user_id,
            &model_ref.raw,
            usage.prompt_tokens,
            usage.completion_tokens,
            cost,
            &request_id,
        )?;
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
) -> GatewayResult<impl futures_util::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send + 'static>
{
    let request_id = Uuid::new_v4().to_string();
    let user_id = crate::gateway::auth::resolve_user_id(&auth, body.user.as_deref())?;
    let model = body.completion.model.clone();
    preflight(&db, &auth, &user_id, &model)?;

    let options = chat_options_from_request(&body.completion, &request_id);
    let tool_specs = tool_specs_from_request(&body.completion);
    let tool_refs = tool_specs.as_deref();

    let (byte_stream, model_ref) = proxy_chat_stream(
        http,
        credentials.as_ref(),
        &body.completion.messages,
        tool_refs,
        &options,
        &request_id,
    )
    .await
    .map_err(chat_error_to_gateway)?;

    let is_anthropic = model_ref.provider == ProviderId::Anthropic;
    let key_id = auth.key_id.clone();
    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let mut stream = byte_stream;
        let mut parser = SseParser::new();
        let mut round_usage: Option<proto::Usage> = None;
        let mut anthropic_acc =
            crate::providers::anthropic_stream::AnthropicStreamAccumulator::new();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if tx.send(Ok(bytes.clone())).is_err() {
                        return;
                    }
                    let text = String::from_utf8_lossy(&bytes);
                    if let Ok(events) = parser.push_str(&text) {
                        for event in events {
                            let data = event.data.trim();
                            if data == "[DONE]" {
                                continue;
                            }
                            if is_anthropic {
                                let _ = anthropic_acc.apply_sse_data(data);
                            } else if let Ok(chunk) =
                                serde_json::from_str::<ChatCompletionChunk>(data)
                                && let Some(u) = chunk.usage
                            {
                                round_usage = Some(proto::Usage {
                                    prompt_tokens: u.prompt_tokens,
                                    completion_tokens: u.completion_tokens,
                                    total_tokens: u.total_tokens,
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e.to_string())));
                    return;
                }
            }
        }

        let usage = if is_anthropic {
            anthropic_acc.into_round_outcome().usage
        } else {
            round_usage
        };

        if let Some(usage) = usage {
            let cost = estimate_model_call_cost_usd(&model, &usage);
            let _ = db.record_usage(
                key_id.as_deref(),
                &user_id,
                &model_ref.raw,
                usage.prompt_tokens,
                usage.completion_tokens,
                cost,
                &request_id,
            );
        }
    });

    Ok(UnboundedReceiverStream::new(rx))
}

/// List models visible to the authenticated caller.
#[must_use]
pub fn models_for_auth(auth: &AuthContext) -> Vec<serde_json::Value> {
    if is_unrestricted(&auth.allowed_models) {
        return vec![serde_json::json!({
            "id": "*",
            "object": "model",
            "owned_by": "superglue-gateway"
        })];
    }
    auth.allowed_models
        .as_ref()
        .map(|patterns| {
            patterns
                .iter()
                .map(|id| {
                    serde_json::json!({
                        "id": id,
                        "object": "model",
                        "owned_by": "superglue-gateway"
                    })
                })
                .collect()
        })
        .unwrap_or_default()
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
