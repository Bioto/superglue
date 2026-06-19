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
}

/// Result of one streamed LLM round (before tool dispatch).
#[derive(Debug, Clone)]
pub struct StreamRoundOutcome {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<proto::Usage>,
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
    pub stream: bool,
    pub options: &'a crate::chat::ChatOptions,
}

pub trait LlmProvider: Send + Sync {
    fn provider_id(&self) -> ProviderId;

    fn build_chat_request(&self, ctx: &ProviderRequestContext<'_>) -> ProviderRequest;

    fn parse_chat_response(&self, json: &Value) -> Result<NormalizedCompletion, ProviderParseError>;

    fn supports_file_upload(&self) -> bool {
        false
    }
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
        Box::new(super::openai_compat::OpenAiCompatProvider::new(model_ref.provider))
    } else {
        Box::new(super::anthropic::AnthropicProvider::new())
    }
}
