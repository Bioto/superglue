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
            warn!(
                provider = provider,
                "unknown provider prefix; defaulting to openai"
            );
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

/// Model string for the HTTP request body.
///
/// Gateway/proxy base URLs receive `provider:model` (`raw`) when present; direct
/// provider APIs receive the bare model id.
#[must_use]
pub fn wire_model_id(model_ref: &ModelRef, base_url: &str, provider: ProviderId) -> String {
    if base_url != provider.default_base_url() && model_ref.raw.contains(':') {
        model_ref.raw.clone()
    } else {
        model_ref.model.clone()
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

    #[test]
    fn parses_groq_gpt_oss() {
        let r = parse_model_ref("groq:openai/gpt-oss-120b");
        assert_eq!(r.provider, ProviderId::Groq);
        assert_eq!(r.model, "openai/gpt-oss-120b");
    }

    #[test]
    fn parses_runinfra_deepseek() {
        let r = parse_model_ref("runinfra:deepseek-v4-flash");
        assert_eq!(r.provider, ProviderId::RunInfra);
        assert_eq!(r.model, "deepseek-v4-flash");
    }

    #[test]
    fn parses_vercel_slash_model() {
        let r = parse_model_ref("vercel:anthropic/claude-opus-5");
        assert_eq!(r.provider, ProviderId::Vercel);
        assert_eq!(r.model, "anthropic/claude-opus-5");
    }

    #[test]
    fn parses_typesafe_jev() {
        let r = parse_model_ref("typesafe:jev-latest");
        assert_eq!(r.provider, ProviderId::TypeSafe);
        assert_eq!(r.model, "jev-latest");
    }

    #[test]
    fn wire_model_uses_prefix_for_gateway_base_url() {
        let model_ref = parse_model_ref("openai:gpt-4o-mini");
        let wired = wire_model_id(
            &model_ref,
            "https://gateway.example.com",
            ProviderId::OpenAi,
        );
        assert_eq!(wired, "openai:gpt-4o-mini");
    }

    #[test]
    fn wire_model_uses_bare_for_direct_openai() {
        let model_ref = parse_model_ref("openai:gpt-4o-mini");
        let wired = wire_model_id(
            &model_ref,
            ProviderId::OpenAi.default_base_url(),
            ProviderId::OpenAi,
        );
        assert_eq!(wired, "gpt-4o-mini");
    }
}
