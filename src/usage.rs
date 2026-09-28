//! Canonical helpers for [`crate::proto::Usage`] construction and aggregation.

use crate::proto;

/// Provider-neutral token usage breakdown before normalization into [`proto::Usage`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UsageBreakdown {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
    pub reasoning_tokens: Option<u32>,
}

impl UsageBreakdown {
    #[must_use]
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: None,
            cached_tokens: None,
            reasoning_tokens: None,
        }
    }
}

/// Build canonical [`proto::Usage`] from a provider-neutral breakdown.
#[must_use]
pub fn usage_from_breakdown(b: UsageBreakdown) -> proto::Usage {
    let total_tokens = b
        .total_tokens
        .unwrap_or_else(|| b.prompt_tokens.saturating_add(b.completion_tokens));
    proto::Usage {
        prompt_tokens: b.prompt_tokens,
        completion_tokens: b.completion_tokens,
        total_tokens,
        cached_tokens: b.cached_tokens.filter(|&n| n > 0),
        reasoning_tokens: b.reasoning_tokens.filter(|&n| n > 0),
        cost_usd: None,
    }
}

/// Map OpenAI-compatible usage, including billed `cost` when present.
#[must_use]
pub fn usage_from_compat(u: &crate::openai::Usage) -> proto::Usage {
    let mut usage = usage_from_breakdown(breakdown_from_compat_usage(u));
    usage.cost_usd = crate::costing::sanitize_billed_cost_usd(u.cost);
    usage
}

/// JSON object used by language bindings.
#[must_use]
pub fn usage_to_json(u: &proto::Usage) -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "cost_usd": u.cost_usd,
    })
}

/// Map a provider HTTP usage object (OpenAI-compatible shape) into [`UsageBreakdown`].
pub(crate) fn breakdown_from_compat_usage(u: &crate::openai::Usage) -> UsageBreakdown {
    UsageBreakdown {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: Some(u.total_tokens),
        cached_tokens: cached_tokens_from_compat(u),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens)
            .filter(|&n| n > 0),
    }
}

#[must_use]
pub fn accumulate_usage(prev: Option<&proto::Usage>, round: &proto::Usage) -> proto::Usage {
    let Some(prev) = prev else {
        return round.clone();
    };
    proto::Usage {
        prompt_tokens: prev.prompt_tokens.saturating_add(round.prompt_tokens),
        completion_tokens: prev
            .completion_tokens
            .saturating_add(round.completion_tokens),
        total_tokens: prev.total_tokens.saturating_add(round.total_tokens),
        cached_tokens: sum_optional(prev.cached_tokens, round.cached_tokens),
        reasoning_tokens: sum_optional(prev.reasoning_tokens, round.reasoning_tokens),
        cost_usd: sum_optional_f64(prev.cost_usd, round.cost_usd),
    }
}

fn first_positive(candidates: impl IntoIterator<Item = Option<u32>>) -> Option<u32> {
    candidates.into_iter().flatten().find(|&n| n > 0)
}

/// Cache hits from OpenAI-compatible usage, including DeepSeek and aggregator aliases.
#[must_use]
pub fn cached_tokens_from_compat(u: &crate::openai::Usage) -> Option<u32> {
    first_positive([
        u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens),
        u.prompt_cache_hit_tokens,
        u.cache_read_input_tokens,
        u.cached_tokens,
    ])
}

/// Cache hits from a raw usage JSON object.
#[must_use]
pub fn cached_tokens_from_usage_json(u: &serde_json::Value) -> Option<u32> {
    first_positive([
        u.pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        u.pointer("/input_tokens_details/cached_tokens")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        u.get("prompt_cache_hit_tokens")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        u.get("cache_read_input_tokens")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        u.get("native_tokens_cached")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
        u.get("cached_tokens")
            .and_then(|x| x.as_u64())
            .map(|n| n as u32),
    ])
}

fn sum_optional(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
    }
}

fn sum_optional_f64(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => Some(x + y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_sums_token_fields() {
        let a = proto::Usage {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            cached_tokens: Some(50),
            reasoning_tokens: None,
            cost_usd: Some(0.01),
        };
        let b = proto::Usage {
            prompt_tokens: 200,
            completion_tokens: 30,
            total_tokens: 230,
            cached_tokens: Some(10),
            reasoning_tokens: Some(5),
            cost_usd: Some(0.02),
        };
        let sum = accumulate_usage(Some(&a), &b);
        assert_eq!(sum.prompt_tokens, 300);
        assert_eq!(sum.completion_tokens, 50);
        assert_eq!(sum.total_tokens, 350);
        assert_eq!(sum.cached_tokens, Some(60));
        assert_eq!(sum.reasoning_tokens, Some(5));
        assert_eq!(sum.cost_usd, Some(0.03));
    }

    #[test]
    fn usage_from_breakdown_preserves_cached_tokens() {
        let proto = usage_from_breakdown(UsageBreakdown {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: Some(15),
            cached_tokens: Some(8),
            reasoning_tokens: None,
        });
        assert_eq!(proto.cached_tokens, Some(8));
        assert_eq!(proto.total_tokens, 15);
    }

    #[test]
    fn breakdown_from_compat_usage_maps_details() {
        let u = crate::openai::Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            completion_tokens_details: None,
            prompt_tokens_details: Some(crate::openai::TokenDetails {
                cached_tokens: 8,
                reasoning_tokens: 0,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            }),
            prompt_cache_hit_tokens: None,
            cache_read_input_tokens: None,
            cached_tokens: None,
            cost: Some(0.0123),
        };
        let proto = usage_from_compat(&u);
        assert_eq!(proto.cost_usd, Some(0.0123));
        let b = breakdown_from_compat_usage(&u);
        assert_eq!(b.cached_tokens, Some(8));
    }

    #[test]
    fn breakdown_from_compat_usage_reads_deepseek_cache_hit_tokens() {
        let u = crate::openai::Usage {
            prompt_tokens: 100,
            completion_tokens: 5,
            total_tokens: 105,
            completion_tokens_details: None,
            prompt_tokens_details: None,
            prompt_cache_hit_tokens: Some(80),
            cache_read_input_tokens: None,
            cached_tokens: None,
            cost: None,
        };
        assert_eq!(breakdown_from_compat_usage(&u).cached_tokens, Some(80));
    }

    #[test]
    fn cached_tokens_from_usage_json_reads_provider_aliases() {
        let deepseek = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_cache_hit_tokens": 80
        });
        assert_eq!(cached_tokens_from_usage_json(&deepseek), Some(80));
        let anthropic = serde_json::json!({ "cache_read_input_tokens": 40 });
        assert_eq!(cached_tokens_from_usage_json(&anthropic), Some(40));
        let openrouter = serde_json::json!({ "native_tokens_cached": 12 });
        assert_eq!(cached_tokens_from_usage_json(&openrouter), Some(12));
    }
}
