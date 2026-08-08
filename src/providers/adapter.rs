//! Provider adapter trait and normalized completion types.

use serde_json::Value;

use crate::openai::{ChatMessage, ToolCall};
use crate::proto;
use crate::tools::ToolSpec;

use super::credentials::ProviderCredentials;
use super::model_ref::ModelRef;
use super::provider_id::ProviderId;
use super::rate_limit::RateLimitKey;

/// Unified completion result across providers.
#[derive(Debug, Clone)]
pub struct NormalizedCompletion {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<proto::Usage>,
    pub finish_reason: Option<String>,
    /// Provider-native stop reason when different from OpenAI finish_reason.
    pub stop_reason: Option<String>,
    /// Provider-native assistant blocks for round-trip (e.g. Anthropic thinking).
    pub provider_blocks: Option<Vec<Value>>,
}

/// Result of one streamed LLM round (before tool dispatch).
#[derive(Debug, Clone)]
pub struct StreamRoundOutcome {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<proto::Usage>,
    /// Provider-native assistant blocks for round-trip (e.g. Anthropic thinking).
    pub provider_blocks: Option<Vec<Value>>,
}

#[derive(Debug, Clone)]
pub struct ProviderRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub rate_limit_key: RateLimitKey,
    pub model_ref: ModelRef,
    pub bare_model: String,
}

/// Context for building a provider request.
pub struct ProviderRequestContext<'a> {
    pub model_ref: &'a ModelRef,
    pub credentials: &'a ProviderCredentials,
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    pub chat_tools: Option<&'a [crate::openai::ChatTool]>,
    pub stream: bool,
    pub options: &'a crate::chat::ChatOptions,
}

/// Context for building a Responses API (`POST /v1/responses`) request.
pub struct ProviderResponsesContext<'a> {
    pub model_ref: &'a ModelRef,
    pub credentials: &'a ProviderCredentials,
    pub body: &'a Value,
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    pub stream: bool,
    pub options: &'a crate::chat::ChatOptions,
}

/// Normalized Responses API round result (OpenAI-shaped output).
#[derive(Debug, Clone)]
pub struct NormalizedResponse {
    pub id: String,
    pub output: Value,
    pub usage: Option<proto::Usage>,
    /// Provider-native assistant blocks for round-trip (e.g. Anthropic thinking).
    pub provider_blocks: Option<Vec<Value>>,
}

pub trait LlmProvider: Send + Sync {
    fn provider_id(&self) -> ProviderId;

    fn build_chat_request(&self, ctx: &ProviderRequestContext<'_>) -> ProviderRequest;

    fn parse_chat_response(&self, json: &Value)
    -> Result<NormalizedCompletion, ProviderParseError>;

    fn supports_file_upload(&self) -> bool {
        false
    }

    /// Whether `previous_response_id` chaining is supported on the Responses path.
    fn supports_previous_response_id(&self) -> bool {
        matches!(
            self.provider_id(),
            ProviderId::OpenAi | ProviderId::Xai
        )
    }

    /// Whether `prompt_cache_key` is supported on the Responses path.
    fn supports_prompt_cache_key(&self) -> bool {
        matches!(self.provider_id(), ProviderId::OpenAi | ProviderId::Xai)
    }

    /// Build a provider HTTP request for one Responses API round.
    fn build_responses_request(&self, ctx: &ProviderResponsesContext<'_>) -> ProviderRequest;

    /// Parse a provider HTTP JSON body into OpenAI Responses-shaped output.
    fn parse_responses_response(
        &self,
        json: &Value,
    ) -> Result<NormalizedResponse, ProviderParseError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderParseError {
    #[error("failed to parse provider response: {0}")]
    InvalidResponse(String),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

pub fn rate_limit_key_for(
    model_ref: &ModelRef,
    credentials: &ProviderCredentials,
) -> Result<RateLimitKey, super::credentials::CredentialsError> {
    let key = credentials.key_for(model_ref.provider)?;
    Ok(RateLimitKey {
        provider: model_ref.provider,
        key_id: super::credentials::api_key_id(&key),
    })
}

pub fn resolve_provider(model_ref: &ModelRef) -> Box<dyn LlmProvider> {
    if model_ref.provider.uses_openai_compat() {
        Box::new(super::openai_compat::OpenAiCompatProvider::new(
            model_ref.provider,
        ))
    } else {
        Box::new(super::anthropic::AnthropicProvider::new())
    }
}
