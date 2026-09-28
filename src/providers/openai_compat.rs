//! OpenAI-compatible chat API (OpenAI, Groq, xAI, OpenRouter, RunInfra, Vercel).

use std::collections::HashMap;

use secrecy::ExposeSecret;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Value, json};

use crate::chat::reasoning::{normalize_reasoning_effort_str, supports_chat_sampling_params};
use crate::http::join_base_url;
use crate::openai::{
    ChatCompletionResponse, ChatMessage, ChatTool, ResponseFormat, StopSequence, StreamOptions,
    ToolChoice,
};

use super::adapter::{
    LlmProvider, NormalizedCompletion, NormalizedResponse, ProviderParseError, ProviderRequest,
    ProviderRequestContext, ProviderResponsesContext, rate_limit_key_for,
};
use super::model_ref::wire_model_id;
use super::provider_id::ProviderId;

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
        append_client_request_id(
            self.provider,
            &mut headers,
            ctx.options.request_id.as_deref(),
        );

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
        let reasoning_effort =
            resolve_reasoning_effort(self.provider, &ctx.model_ref.model, options);

        let stream_options = if ctx.stream {
            Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            })
        } else {
            None
        };

        let mut extra = HashMap::new();
        if self.supports_prompt_cache_key()
            && let Some(key) = &options.prompt_cache_key
        {
            extra.insert("prompt_cache_key".to_string(), json!(key));
        }
        if self.provider.exposes_generation_cost() {
            extra.insert("usage".to_string(), json!({ "include": true }));
        }
        let mut max_completion_tokens = options.max_completion_tokens;
        if self.provider.uses_legacy_max_tokens()
            && let Some(max) = max_completion_tokens.take()
        {
            extra.insert("max_tokens".to_string(), json!(max));
        }

        let wire_model = wire_model_id(ctx.model_ref, &base_url, ctx.model_ref.provider);
        let sampling = supports_chat_sampling_params(&ctx.model_ref.model)
            && supports_chat_sampling_params(&ctx.model_ref.raw);

        let req_ref = ChatCompletionRequestRef {
            model: &wire_model,
            messages: ctx.messages,
            tools: tools_ref,
            tool_choice: options.tool_choice.as_ref(),
            parallel_tool_calls: options.parallel_tool_calls,
            temperature: sampling.then_some(options.temperature).flatten(),
            top_p: sampling.then_some(options.top_p).flatten(),
            n: options.n,
            max_completion_tokens,
            presence_penalty: sampling.then_some(options.presence_penalty).flatten(),
            frequency_penalty: sampling.then_some(options.frequency_penalty).flatten(),
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
        let usage = response.usage.as_ref().map(crate::usage::usage_from_compat);
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
        append_client_request_id(
            self.provider,
            &mut headers,
            ctx.options.request_id.as_deref(),
        );

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
            let cached = crate::usage::cached_tokens_from_usage_json(u);
            let mut usage = crate::usage::usage_from_breakdown(crate::usage::UsageBreakdown {
                prompt_tokens: input,
                completion_tokens: output_tokens,
                total_tokens: Some(total),
                cached_tokens: cached,
                reasoning_tokens: None,
            });
            usage.cost_usd = crate::costing::billed_cost_usd_from_usage_json(u);
            Some(usage)
        });
        Ok(NormalizedResponse {
            id,
            output,
            usage,
            provider_blocks: None,
        })
    }

    fn supports_previous_response_id(&self) -> bool {
        matches!(self.provider, ProviderId::OpenAi | ProviderId::Xai)
    }

    fn supports_file_upload(&self) -> bool {
        matches!(self.provider, ProviderId::OpenAi | ProviderId::Xai)
    }
}

fn resolve_reasoning_effort(
    provider: ProviderId,
    model: &str,
    options: &crate::chat::ChatOptions,
) -> Option<String> {
    let effort = options.reasoning_effort.as_ref()?;
    normalize_reasoning_effort_str(model, effort).or_else(|| {
        provider
            .passthrough_reasoning_effort()
            .then(|| effort.clone())
    })
}

fn append_client_request_id(
    provider: ProviderId,
    headers: &mut Vec<(String, String)>,
    request_id: Option<&str>,
) {
    if !provider.requires_client_request_id() {
        return;
    }
    let id = request_id
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    headers.push(("X-Client-Request-Id".to_string(), id));
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
    fn groq_chat_request_omits_prompt_cache_key() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::Groq, "gsk-test");
        let model_ref = ModelRef {
            provider: ProviderId::Groq,
            model: "openai/gpt-oss-120b".into(),
            raw: "groq:openai/gpt-oss-120b".into(),
        };
        let options = ChatOptions {
            prompt_cache_key: Some("session-abc".into()),
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
        let req = OpenAiCompatProvider::new(ProviderId::Groq).build_chat_request(&ctx);
        assert!(req.body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn openrouter_chat_request_asks_for_usage_details() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::OpenRouter, "or-test");
        let model_ref = ModelRef {
            provider: ProviderId::OpenRouter,
            model: "deepseek/deepseek-v4.1-flash".into(),
            raw: "openrouter:deepseek/deepseek-v4.1-flash".into(),
        };
        let options = ChatOptions::default();
        let ctx = ProviderRequestContext {
            model_ref: &model_ref,
            credentials: &creds,
            messages: &[ChatMessage::text("user", "hi")],
            tools: None,
            chat_tools: None,
            stream: false,
            options: &options,
        };
        let req = OpenAiCompatProvider::new(ProviderId::OpenRouter).build_chat_request(&ctx);
        assert_eq!(req.body.pointer("/usage/include"), Some(&json!(true)));
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

    fn runinfra_chat_ctx<'a>(
        creds: &'a ProviderCredentials,
        model_ref: &'a ModelRef,
        messages: &'a [ChatMessage],
        options: &'a ChatOptions,
    ) -> ProviderRequestContext<'a> {
        ProviderRequestContext {
            model_ref,
            credentials: creds,
            messages,
            tools: None,
            chat_tools: None,
            stream: false,
            options,
        }
    }

    #[test]
    fn runinfra_chat_request_sets_client_request_id_and_max_tokens() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::RunInfra, "ri-test");
        let model_ref = ModelRef {
            provider: ProviderId::RunInfra,
            model: "deepseek-v4-flash".into(),
            raw: "runinfra:deepseek-v4-flash".into(),
        };
        let options = ChatOptions {
            max_completion_tokens: Some(16384),
            reasoning_effort: Some("max".into()),
            request_id: Some("req-123".into()),
            ..Default::default()
        };
        let messages = [ChatMessage::text("user", "Hello")];
        let req = OpenAiCompatProvider::new(ProviderId::RunInfra)
            .build_chat_request(&runinfra_chat_ctx(&creds, &model_ref, &messages, &options));
        assert!(req.url.ends_with("/v1/chat/completions"));
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "X-Client-Request-Id" && v == "req-123")
        );
        assert_eq!(
            req.body.get("max_tokens").and_then(|v| v.as_u64()),
            Some(16384)
        );
        assert!(req.body.get("max_completion_tokens").is_none());
        assert_eq!(
            req.body.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("max")
        );
        assert_eq!(
            req.body.get("model").and_then(|v| v.as_str()),
            Some("deepseek-v4-flash")
        );
    }

    #[test]
    fn runinfra_generates_client_request_id_when_missing() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::RunInfra, "ri-test");
        let model_ref = ModelRef {
            provider: ProviderId::RunInfra,
            model: "deepseek-v4-flash".into(),
            raw: "runinfra:deepseek-v4-flash".into(),
        };
        let options = ChatOptions::default();
        let messages = [ChatMessage::text("user", "Hello")];
        let req = OpenAiCompatProvider::new(ProviderId::RunInfra)
            .build_chat_request(&runinfra_chat_ctx(&creds, &model_ref, &messages, &options));
        let header = req
            .headers
            .iter()
            .find(|(k, _)| k == "X-Client-Request-Id")
            .map(|(_, v)| v.as_str());
        assert!(header.is_some_and(|v| !v.is_empty()));
    }

    #[test]
    fn vercel_chat_request_uses_openai_compat_and_bare_model() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::Vercel, "vck-test");
        let model_ref = ModelRef {
            provider: ProviderId::Vercel,
            model: "anthropic/claude-opus-5".into(),
            raw: "vercel:anthropic/claude-opus-5".into(),
        };
        let options = ChatOptions {
            max_completion_tokens: Some(4096),
            reasoning_effort: Some("medium".into()),
            ..Default::default()
        };
        let messages = [ChatMessage::text("user", "Hello")];
        let req = OpenAiCompatProvider::new(ProviderId::Vercel).build_chat_request(
            &ProviderRequestContext {
                model_ref: &model_ref,
                credentials: &creds,
                messages: &messages,
                tools: None,
                chat_tools: None,
                stream: false,
                options: &options,
            },
        );
        assert_eq!(req.url, "https://ai-gateway.vercel.sh/v1/chat/completions");
        assert!(
            req.headers
                .iter()
                .any(|(k, v)| k == "Authorization" && v == "Bearer vck-test")
        );
        assert!(!req.headers.iter().any(|(k, _)| k == "X-Client-Request-Id"));
        assert_eq!(
            req.body
                .get("max_completion_tokens")
                .and_then(|v| v.as_u64()),
            Some(4096)
        );
        assert!(req.body.get("max_tokens").is_none());
        assert_eq!(
            req.body.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("medium")
        );
        assert_eq!(
            req.body.get("model").and_then(|v| v.as_str()),
            Some("anthropic/claude-opus-5")
        );
    }

    fn openai_chat_ctx<'a>(
        creds: &'a ProviderCredentials,
        model_ref: &'a ModelRef,
        messages: &'a [ChatMessage],
        options: &'a ChatOptions,
    ) -> ProviderRequestContext<'a> {
        ProviderRequestContext {
            model_ref,
            credentials: creds,
            messages,
            tools: None,
            chat_tools: None,
            stream: false,
            options,
        }
    }

    #[test]
    fn gpt5_chat_request_omits_temperature() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::OpenAi, "sk-test");
        let model_ref = ModelRef {
            provider: ProviderId::OpenAi,
            model: "gpt-5.6-luna".into(),
            raw: "openai:gpt-5.6-luna".into(),
        };
        let options = ChatOptions {
            temperature: Some(0.0),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            ..Default::default()
        };
        let messages = [ChatMessage::text("user", "hi")];
        let req = OpenAiCompatProvider::new(ProviderId::OpenAi)
            .build_chat_request(&openai_chat_ctx(&creds, &model_ref, &messages, &options));
        assert!(req.body.get("temperature").is_none());
        assert!(req.body.get("top_p").is_none());
        assert!(req.body.get("presence_penalty").is_none());
        assert!(req.body.get("frequency_penalty").is_none());
    }

    #[test]
    fn gpt4o_chat_request_keeps_temperature() {
        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::OpenAi, "sk-test");
        let model_ref = ModelRef {
            provider: ProviderId::OpenAi,
            model: "gpt-4o".into(),
            raw: "openai:gpt-4o".into(),
        };
        let options = ChatOptions {
            temperature: Some(0.0),
            ..Default::default()
        };
        let messages = [ChatMessage::text("user", "hi")];
        let req = OpenAiCompatProvider::new(ProviderId::OpenAi)
            .build_chat_request(&openai_chat_ctx(&creds, &model_ref, &messages, &options));
        assert_eq!(
            req.body.get("temperature").and_then(|v| v.as_f64()),
            Some(0.0)
        );
    }
}
