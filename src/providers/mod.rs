//! Multi-provider routing (`provider:model`), credentials, and per-API-key rate limits.

mod adapter;
mod anthropic;
pub mod anthropic_stream;
mod credentials;
mod model_ref;
mod openai_compat;
mod provider_id;
mod rate_limit;

pub use adapter::{
    LlmProvider, NormalizedCompletion, NormalizedResponse, ProviderParseError, ProviderRequest,
    ProviderRequestContext, ProviderResponsesContext, StreamRoundOutcome, rate_limit_key_for,
    resolve_provider,
};
pub use credentials::{ApiKeyId, CredentialsError, ProviderCredentials, api_key_id};
pub use model_ref::{ModelRef, parse_model_ref};
pub use provider_id::{ProviderId, UnknownProvider};
pub use rate_limit::{RateLimitKey, RateLimitRegistry};
