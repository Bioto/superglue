//! Known provider identifiers (`openai`, `anthropic`, `xai`, `groq`, `openrouter`, `runinfra`, `vercel`, `typesafe`).

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderId {
    OpenAi,
    Anthropic,
    Xai,
    Groq,
    OpenRouter,
    RunInfra,
    Vercel,
    TypeSafe,
}

#[derive(Debug, Error)]
#[error("unknown provider '{0}'")]
pub struct UnknownProvider(pub String);

impl ProviderId {
    pub const ALL: [ProviderId; 8] = [
        ProviderId::OpenAi,
        ProviderId::Anthropic,
        ProviderId::Xai,
        ProviderId::Groq,
        ProviderId::OpenRouter,
        ProviderId::RunInfra,
        ProviderId::Vercel,
        ProviderId::TypeSafe,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "openai",
            ProviderId::Anthropic => "anthropic",
            ProviderId::Xai => "xai",
            ProviderId::Groq => "groq",
            ProviderId::OpenRouter => "openrouter",
            ProviderId::RunInfra => "runinfra",
            ProviderId::Vercel => "vercel",
            ProviderId::TypeSafe => "typesafe",
        }
    }

    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "OpenAI",
            ProviderId::Anthropic => "Anthropic",
            ProviderId::Xai => "xAI",
            ProviderId::Groq => "Groq",
            ProviderId::OpenRouter => "OpenRouter",
            ProviderId::RunInfra => "RunInfra",
            ProviderId::Vercel => "Vercel",
            ProviderId::TypeSafe => "TypeSafe",
        }
    }

    #[must_use]
    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "https://api.openai.com",
            ProviderId::Anthropic => "https://api.anthropic.com",
            ProviderId::Xai => "https://api.x.ai",
            ProviderId::Groq => "https://api.groq.com/openai",
            ProviderId::OpenRouter => "https://openrouter.ai/api",
            ProviderId::RunInfra => "https://api.runinfra.ai",
            ProviderId::Vercel => "https://ai-gateway.vercel.sh",
            ProviderId::TypeSafe => "https://api.typesafe.ai",
        }
    }

    #[must_use]
    pub fn env_var_for_key(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "OPENAI_API_KEY",
            ProviderId::Anthropic => "ANTHROPIC_API_KEY",
            ProviderId::Xai => "XAI_API_KEY",
            ProviderId::Groq => "GROQ_API_KEY",
            ProviderId::OpenRouter => "OPENROUTER_API_KEY",
            ProviderId::RunInfra => "RUNINFRA_GATEWAY_KEY",
            ProviderId::Vercel => "VERCEL_GATEWAY_KEY",
            ProviderId::TypeSafe => "TYPESAFE_API_KEY",
        }
    }

    #[must_use]
    pub fn uses_openai_compat(self) -> bool {
        matches!(
            self,
            ProviderId::OpenAi
                | ProviderId::Xai
                | ProviderId::Groq
                | ProviderId::OpenRouter
                | ProviderId::RunInfra
                | ProviderId::Vercel
        )
    }

    /// Whether the provider exposes OpenAI-compatible `/v1/responses`.
    #[must_use]
    pub fn uses_responses_api(self) -> bool {
        self.uses_openai_compat()
    }

    /// Aggregators that host many model families and should forward `reasoning_effort`.
    #[must_use]
    pub fn passthrough_reasoning_effort(self) -> bool {
        matches!(
            self,
            ProviderId::OpenRouter | ProviderId::RunInfra | ProviderId::Vercel
        )
    }

    /// RunInfra requires `X-Client-Request-Id` on chat requests.
    #[must_use]
    pub fn requires_client_request_id(self) -> bool {
        matches!(self, ProviderId::RunInfra)
    }

    /// RunInfra's chat API accepts `max_tokens` rather than `max_completion_tokens`.
    #[must_use]
    pub fn uses_legacy_max_tokens(self) -> bool {
        matches!(self, ProviderId::RunInfra)
    }

    /// Whether the provider exposes billed USD on `GET /v1/generation`.
    #[must_use]
    pub fn exposes_generation_cost(self) -> bool {
        matches!(self, ProviderId::OpenRouter | ProviderId::Vercel)
    }

    /// Whether the provider exposes the TypeSafe System One evaluation API.
    #[must_use]
    pub fn uses_system_one(self) -> bool {
        matches!(self, ProviderId::TypeSafe)
    }

    /// Whether the provider implements chat completions or Responses.
    #[must_use]
    pub fn supports_chat(self) -> bool {
        self.uses_openai_compat() || matches!(self, ProviderId::Anthropic)
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderId {
    type Err = UnknownProvider;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Ok(ProviderId::OpenAi),
            "anthropic" => Ok(ProviderId::Anthropic),
            "xai" => Ok(ProviderId::Xai),
            "groq" => Ok(ProviderId::Groq),
            "openrouter" => Ok(ProviderId::OpenRouter),
            "runinfra" => Ok(ProviderId::RunInfra),
            "vercel" => Ok(ProviderId::Vercel),
            "typesafe" => Ok(ProviderId::TypeSafe),
            other => Err(UnknownProvider(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_runinfra_prefix() {
        assert_eq!(
            ProviderId::from_str("runinfra").unwrap(),
            ProviderId::RunInfra
        );
        assert_eq!(ProviderId::RunInfra.as_str(), "runinfra");
        assert_eq!(
            ProviderId::RunInfra.env_var_for_key(),
            "RUNINFRA_GATEWAY_KEY"
        );
        assert_eq!(
            ProviderId::RunInfra.default_base_url(),
            "https://api.runinfra.ai"
        );
        assert!(ProviderId::RunInfra.uses_openai_compat());
        assert!(ProviderId::RunInfra.requires_client_request_id());
        assert!(ProviderId::RunInfra.uses_legacy_max_tokens());
    }

    #[test]
    fn parses_vercel_prefix() {
        assert_eq!(ProviderId::from_str("vercel").unwrap(), ProviderId::Vercel);
        assert_eq!(ProviderId::Vercel.as_str(), "vercel");
        assert_eq!(ProviderId::Vercel.env_var_for_key(), "VERCEL_GATEWAY_KEY");
        assert_eq!(
            ProviderId::Vercel.default_base_url(),
            "https://ai-gateway.vercel.sh"
        );
        assert!(ProviderId::Vercel.uses_openai_compat());
        assert!(ProviderId::Vercel.passthrough_reasoning_effort());
        assert!(!ProviderId::Vercel.requires_client_request_id());
        assert!(!ProviderId::Vercel.uses_legacy_max_tokens());
        assert!(ProviderId::Vercel.exposes_generation_cost());
        assert!(ProviderId::OpenRouter.exposes_generation_cost());
        assert!(!ProviderId::OpenAi.exposes_generation_cost());
    }

    #[test]
    fn parses_typesafe_prefix() {
        assert_eq!(
            ProviderId::from_str("typesafe").unwrap(),
            ProviderId::TypeSafe
        );
        assert_eq!(ProviderId::TypeSafe.as_str(), "typesafe");
        assert_eq!(ProviderId::TypeSafe.env_var_for_key(), "TYPESAFE_API_KEY");
        assert_eq!(
            ProviderId::TypeSafe.default_base_url(),
            "https://api.typesafe.ai"
        );
        assert!(!ProviderId::TypeSafe.uses_openai_compat());
        assert!(!ProviderId::TypeSafe.uses_responses_api());
        assert!(ProviderId::TypeSafe.uses_system_one());
        assert!(!ProviderId::TypeSafe.supports_chat());
        assert!(ProviderId::OpenAi.supports_chat());
        assert!(ProviderId::Anthropic.supports_chat());
        assert!(!ProviderId::OpenAi.uses_system_one());
    }
}
