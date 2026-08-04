//! OpenAI-compatible chat API (OpenAI, Groq, xAI).

use std::collections::HashMap;

use secrecy::ExposeSecret;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, json};

use crate::chat::reasoning::normalize_reasoning_effort_str;
use crate::http::join_base_url;
use crate::openai::{
    ChatCompletionResponse, ChatMessage, ChatTool, ResponseFormat, StopSequence, StreamOptions,
    ToolChoice,
};

use super::adapter::{
    LlmProvider, NormalizedCompletion, NormalizedResponse, ProviderParseError, ProviderRequest,
    ProviderRequestContext, ProviderResponsesContext, rate_limit_key_for,
};
use super::provider_id::ProviderId;
use super::model_ref::wire_model_id;

#[derive(Debug, Clone, Copy)]
pub struct OpenAiCompatProvider {
    provider: ProviderId,
}

impl OpenAiCompatProvider {
    #[must_use]
    pub fn new(provider: ProviderId) -> Self {
        Self { provider }
    }
}

impl LlmProvider for OpenAiCompatProvider {
    fn provider_id(&self) -> ProviderId {
        self.provider
    }

    fn build_chat_request(&self, ctx: &ProviderRequestContext<'_>) -> ProviderRequest {
        let base_url = ctx.credentials.base_url_for(ctx.model_ref.provider);
        let url = join_base_url(&base_url, "/v1/chat/completions");
        let api_key = ctx
            .credentials
            .key_for(ctx.model_ref.provider)
            .expect("credentials checked before build");
        let auth = format!("Bearer {}", api_key.expose_secret());
        let mut headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if ctx.model_ref.provider == ProviderId::Xai
            && let Some(key) = &ctx.options.prompt_cache_key
        {
            headers.push(("x-grok-conv-id".to_string(), key.clone()));
        }

        let tools_owned: Option<Vec<ChatTool>> = ctx.chat_tools.map(|t| t.to_vec()).or_else(|| {
            ctx.tools.map(|specs| {
                specs
                    .iter()
                    .map(|s| ChatTool::from(s.clone()))
                    .collect::<Vec<ChatTool>>()
            })
        });
        let tools_ref: Option<&[ChatTool]> = tools_owned.as_deref().or(ctx.chat_tools);

        let options = ctx.options;
        let reasoning_effort = options
            .reasoning_effort
            .as_ref()
            .and_then(|e| normalize_reasoning_effort_str(&ctx.model_ref.model, e));

        let stream_options = if ctx.stream {
            Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            })
        } else {
            None
        };

        let mut extra = HashMap::new();
        if let Some(key) = &options.prompt_cache_key {
            extra.insert("prompt_cache_key".to_string(), json!(key));
        }

        let wire_model = wire_model_id(ctx.model_ref, &base_url, ctx.model_ref.provider);

        let req_ref = ChatCompletionRequestRef {
            model: &wire_model,
            messages: ctx.messages,
            tools: tools_ref,
            tool_choice: options.tool_choice.as_ref(),
            parallel_tool_calls: options.parallel_tool_calls,
            temperature: options.temperature,
            top_p: options.top_p,
            n: options.n,
            max_completion_tokens: options.max_completion_tokens,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            stop: options.stop.as_ref(),
            response_format: options.response_format.as_ref(),
            logprobs: options.logprobs,
            top_logprobs: options.top_logprobs,
            seed: options.seed,
            store: options.store,
            service_tier: options.service_tier.as_deref(),
            stream: ctx.stream.then_some(true),
            stream_options: stream_options.as_ref(),
            reasoning_effort: reasoning_effort.as_deref(),
            extra: &extra,
        };

        let mut body = serde_json::to_value(&req_ref).expect("ChatCompletionRequestRef serializes");
        if let Some(extra) = &options.extra_json {
            merge_extra_json(&mut body, extra);
        }

        let rate_limit_key =
            rate_limit_key_for(ctx.model_ref, ctx.credentials).expect("credentials checked");

        ProviderRequest {
            url,
            headers,
            body,
            rate_limit_key,
            model_ref: ctx.model_ref.clone(),
            bare_model: ctx.model_ref.model.clone(),
        }
    }

    fn parse_chat_response(
        &self,
        json: &Value,
    ) -> Result<NormalizedCompletion, ProviderParseError> {
        let response = ChatCompletionResponse::deserialize(json)?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| ProviderParseError::InvalidResponse("no choices".into()))?;
        let msg = &choice.message;
        let content = msg
            .content
            .as_ref()
            .and_then(|c| c.as_text().map(str::to_string));
        let tool_calls = msg.tool_calls.clone().unwrap_or_default();
        let usage = response
            .usage
            .as_ref()
            .map(crate::usage::breakdown_from_compat_usage)
            .map(crate::usage::usage_from_breakdown);
        Ok(NormalizedCompletion {
            content,
            tool_calls,
            usage,
            finish_reason: choice.finish_reason.clone(),
            stop_reason: None,
            provider_blocks: None,
        })
    }

    fn build_responses_request(&self, ctx: &ProviderResponsesContext<'_>) -> ProviderRequest {
        let base_url = ctx.credentials.base_url_for(ctx.model_ref.provider);
        let url = join_base_url(&base_url, "/v1/responses");
        let api_key = ctx
            .credentials
            .key_for(ctx.model_ref.provider)
            .expect("credentials checked before build");
        let auth = format!("Bearer {}", api_key.expose_secret());
        let mut headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if ctx.model_ref.provider == ProviderId::Xai
            && let Some(key) = ctx.options.prompt_cache_key.as_ref()
        {
            headers.push(("x-grok-conv-id".to_string(), key.clone()));
        }

        let mut body = ctx.body.clone();
        if let Some(obj) = body.as_object_mut() {
            let wire_model = wire_model_id(ctx.model_ref, &base_url, ctx.model_ref.provider);
            obj.insert("model".to_string(), json!(wire_model));
            if ctx.stream {
                obj.insert("stream".to_string(), json!(true));
            }
        }

        let rate_limit_key =
            rate_limit_key_for(ctx.model_ref, ctx.credentials).expect("credentials checked");

        ProviderRequest {
            url,
            headers,
            body,
            rate_limit_key,
            model_ref: ctx.model_ref.clone(),
            bare_model: ctx.model_ref.model.clone(),
        }
    }

    fn parse_responses_response(
        &self,
        json: &Value,
    ) -> Result<NormalizedResponse, ProviderParseError> {
        let id = json
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let output = json.get("output").cloned().unwrap_or_else(|| json!([]));
        let usage = json.get("usage").and_then(|u| {
            let input = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let output_tokens = u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let total =
                u.get("total_tokens")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(u64::from(input) + u64::from(output_tokens)) as u32;
            let cached = u
                .pointer("/input_tokens_details/cached_tokens")
                .and_then(|x| x.as_u64())
                .map(|n| n as u32)
                .filter(|&n| n > 0);
            Some(crate::usage::usage_from_breakdown(
                crate::usage::UsageBreakdown {
                    prompt_tokens: input,
                    completion_tokens: output_tokens,
                    total_tokens: Some(total),
                    cached_tokens: cached,
                    reasoning_tokens: None,
                },
            ))
        });
        Ok(NormalizedResponse {
            id,
            output,
            usage,
            provider_blocks: None,
        })
    }

    fn supports_previous_response_id(&self) -> bool {
        self.provider != ProviderId::Groq
    }

    fn supports_file_upload(&self) -> bool {
        self.provider == ProviderId::OpenAi || self.provider == ProviderId::Xai
    }
}

fn merge_extra_json(body: &mut Value, extra: &Value) {
    if let (Value::Object(body_map), Value::Object(extra_map)) = (body, extra) {
        for (k, v) in extra_map {
            body_map.insert(k.clone(), v.clone());
        }
    }
}

#[derive(Serialize)]
struct ChatCompletionRequestRef<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ChatTool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a StopSequence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<&'a StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(flatten)]
    extra: &'a HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatOptions, SystemPromptBlock};
    use crate::openai::ChatMessage;
    use crate::providers::adapter::ProviderResponsesContext;
    use crate::providers::credentials::ProviderCredentials;
    use crate::providers::model_ref::ModelRef;

    #[test]
    fn xai_chat_request_sets_conv_id_header_and_cache_key() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::Xai, "xai-test");
        let model_ref = ModelRef {
            provider: ProviderId::Xai,
            model: "grok-4.5".into(),
            raw: "xai:grok-4.5".into(),
        };
        let options = ChatOptions {
            prompt_cache_key: Some("session-abc".into()),
            system_prompt_blocks: Some(vec![SystemPromptBlock::cached("base")]),
            ..Default::default()
        };
        let ctx = ProviderRequestContext {
            model_ref: &model_ref,
            credentials: &creds,
            messages: &[ChatMessage::text("user", "hi")],
            tools: None,
            chat_tools: None,
            stream: false,
            options: &options,
        };
        let req = OpenAiCompatProvider::new(ProviderId::Xai).build_chat_request(&ctx);
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "x-grok-conv-id" && v == "session-abc")
        );
        assert_eq!(
            req.body.get("prompt_cache_key").and_then(|v| v.as_str()),
            Some("session-abc")
        );
    }

    #[test]
    fn groq_supports_previous_response_id_is_false() {
        let provider = OpenAiCompatProvider::new(ProviderId::Groq);
        assert!(!provider.supports_previous_response_id());
    }

    #[test]
    fn xai_and_openai_support_previous_response_id() {
        assert!(OpenAiCompatProvider::new(ProviderId::Xai).supports_previous_response_id());
        assert!(OpenAiCompatProvider::new(ProviderId::OpenAi).supports_previous_response_id());
    }

    #[test]
    fn xai_responses_request_sets_conv_id_header() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::Xai, "xai-test");
        let model_ref = ModelRef {
            provider: ProviderId::Xai,
            model: "grok-4.5".into(),
            raw: "xai:grok-4.5".into(),
        };
        let options = ChatOptions {
            prompt_cache_key: Some("session-abc".into()),
            ..Default::default()
        };
        let body = json!({
            "model": "xai:grok-4.5",
            "input": "hi",
            "prompt_cache_key": "session-abc"
        });
        let ctx = ProviderResponsesContext {
            model_ref: &model_ref,
            credentials: &creds,
            body: &body,
            messages: &[ChatMessage::text("user", "hi")],
            tools: None,
            stream: false,
            options: &options,
        };
        let req = OpenAiCompatProvider::new(ProviderId::Xai).build_responses_request(&ctx);
        assert!(req.url.ends_with("/v1/responses"));
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "x-grok-conv-id" && v == "session-abc")
        );
        assert_eq!(
            req.body.get("model").and_then(|v| v.as_str()),
            Some("grok-4.5")
        );
    }

    #[test]
    fn responses_request_uses_bare_model_not_prefix() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::Groq, "gsk-test");
        let model_ref = ModelRef {
            provider: ProviderId::Groq,
            model: "llama-3.3-70b-versatile".into(),
            raw: "groq:llama-3.3-70b-versatile".into(),
        };
        let options = ChatOptions::default();
        let body = json!({"model": "groq:llama-3.3-70b-versatile", "input": "hi"});
        let ctx = ProviderResponsesContext {
            model_ref: &model_ref,
            credentials: &creds,
            body: &body,
            messages: &[ChatMessage::text("user", "hi")],
            tools: None,
            stream: false,
            options: &options,
        };
        let req = OpenAiCompatProvider::new(ProviderId::Groq).build_responses_request(&ctx);
        assert_eq!(
            req.body.get("model").and_then(|v| v.as_str()),
            Some("llama-3.3-70b-versatile")
        );
    }
}
