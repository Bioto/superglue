//! Known LLM provider identifiers (`openai`, `anthropic`, `xai`, `groq`).

use std::fmt;
use std::str::FromStr;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderId {
    OpenAi,
    Anthropic,
    Xai,
    Groq,
}

#[derive(Debug, Error)]
#[error("unknown provider '{0}'")]
pub struct UnknownProvider(pub String);

impl ProviderId {
    pub const ALL: [ProviderId; 4] = [
        ProviderId::OpenAi,
        ProviderId::Anthropic,
        ProviderId::Xai,
        ProviderId::Groq,
    ];

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "openai",
            ProviderId::Anthropic => "anthropic",
            ProviderId::Xai => "xai",
            ProviderId::Groq => "groq",
        }
    }

    #[must_use]
    pub fn default_base_url(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "https://api.openai.com",
            ProviderId::Anthropic => "https://api.anthropic.com",
            ProviderId::Xai => "https://api.x.ai",
            ProviderId::Groq => "https://api.groq.com/openai",
        }
    }

    #[must_use]
    pub fn env_var_for_key(self) -> &'static str {
        match self {
            ProviderId::OpenAi => "OPENAI_API_KEY",
            ProviderId::Anthropic => "ANTHROPIC_API_KEY",
            ProviderId::Xai => "XAI_API_KEY",
            ProviderId::Groq => "GROQ_API_KEY",
        }
    }

    #[must_use]
    pub fn uses_openai_compat(self) -> bool {
        matches!(
            self,
            ProviderId::OpenAi | ProviderId::Xai | ProviderId::Groq
        )
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
            other => Err(UnknownProvider(other.to_string())),
        }
    }
}
