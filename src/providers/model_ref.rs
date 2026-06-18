//! `provider:model` parsing (any-llm style).

use std::str::FromStr;
use tracing::warn;

use super::provider_id::ProviderId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRef {
    pub provider: ProviderId,
    /// Bare model id sent to the provider API.
    pub model: String,
    /// Original caller string for telemetry.
    pub raw: String,
}

/// Parse `openai:gpt-4o-mini` or bare `gpt-4o-mini` (defaults to OpenAI).
#[must_use]
pub fn parse_model_ref(s: &str) -> ModelRef {
    let trimmed = s.trim();
    if let Some((provider, model)) = trimmed.split_once(':') {
        let provider = ProviderId::from_str(provider).unwrap_or_else(|_| {
            warn!(provider = provider, "unknown provider prefix; defaulting to openai");
            ProviderId::OpenAi
        });
        return ModelRef {
            provider,
            model: model.to_string(),
            raw: trimmed.to_string(),
        };
    }
    if let Some((provider, model)) = trimmed.split_once('/') {
        warn!(
            model = trimmed,
            "provider/model format is deprecated; use provider:model"
        );
        let provider = ProviderId::from_str(provider).unwrap_or(ProviderId::OpenAi);
        return ModelRef {
            provider,
            model: model.to_string(),
            raw: trimmed.to_string(),
        };
    }
    ModelRef {
        provider: ProviderId::OpenAi,
        model: trimmed.to_string(),
        raw: trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_provider_colon_model() {
        let r = parse_model_ref("anthropic:claude-3-5-sonnet-20241022");
        assert_eq!(r.provider, ProviderId::Anthropic);
        assert_eq!(r.model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn bare_model_defaults_openai() {
        let r = parse_model_ref("gpt-4o-mini");
        assert_eq!(r.provider, ProviderId::OpenAi);
        assert_eq!(r.model, "gpt-4o-mini");
    }
}
