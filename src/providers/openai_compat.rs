//! OpenAI-compatible chat API (OpenAI, Groq, xAI).

use secrecy::ExposeSecret;
use serde_json::Value;

use crate::chat::reasoning::normalize_reasoning_effort_str;
use crate::http::join_base_url;
use crate::openai::{ChatCompletionRequest, ChatTool, StreamOptions};

use super::adapter::{
    LlmProvider, NormalizedCompletion, ProviderParseError, ProviderRequest, ProviderRequestContext,
    rate_limit_key_for,
};
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
        let headers = vec![
            ("Authorization".to_string(), auth),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let tools = ctx.chat_tools.map(|t| t.to_vec()).or_else(|| {
            ctx.tools.map(|specs| {
                specs
                    .iter()
                    .map(|s| ChatTool::from(s.clone()))
                    .collect::<Vec<ChatTool>>()
            })
        });

        let mut req =
            ChatCompletionRequest::new(ctx.model_ref.model.clone(), ctx.messages.to_vec(), tools);
        let options = ctx.options;
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
        if ctx.stream {
            req.stream = Some(true);
            req.stream_options = Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        }
        req.reasoning_effort = options
            .reasoning_effort
            .as_ref()
            .and_then(|e| normalize_reasoning_effort_str(&ctx.model_ref.model, e));

        let mut body = serde_json::to_value(&req).expect("ChatCompletionRequest serializes");
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
        let response: crate::openai::ChatCompletionResponse = serde_json::from_value(json.clone())?;
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
        })
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
