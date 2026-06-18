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
    LlmProvider, NormalizedCompletion, ProviderParseError, ProviderRequest,
    ProviderRequestContext, StreamRoundOutcome, resolve_provider, rate_limit_key_for,
};
pub use credentials::{api_key_id, ApiKeyId, CredentialsError, ProviderCredentials};
pub use model_ref::{parse_model_ref, ModelRef};
pub use provider_id::{ProviderId, UnknownProvider};
pub use rate_limit::{RateLimitKey, RateLimitRegistry};
