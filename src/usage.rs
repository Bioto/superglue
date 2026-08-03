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
    }
}

/// Map a provider HTTP usage object (OpenAI-compatible shape) into [`UsageBreakdown`].
pub(crate) fn breakdown_from_compat_usage(u: &crate::openai::Usage) -> UsageBreakdown {
    UsageBreakdown {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: Some(u.total_tokens),
        cached_tokens: u
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .filter(|&n| n > 0),
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
    }
}

fn sum_optional(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(x), Some(y)) => Some(x.saturating_add(y)),
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
        };
        let b = proto::Usage {
            prompt_tokens: 200,
            completion_tokens: 30,
            total_tokens: 230,
            cached_tokens: Some(10),
            reasoning_tokens: Some(5),
        };
        let sum = accumulate_usage(Some(&a), &b);
        assert_eq!(sum.prompt_tokens, 300);
        assert_eq!(sum.completion_tokens, 50);
        assert_eq!(sum.total_tokens, 350);
        assert_eq!(sum.cached_tokens, Some(60));
        assert_eq!(sum.reasoning_tokens, Some(5));
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
        };
        let b = breakdown_from_compat_usage(&u);
        assert_eq!(b.cached_tokens, Some(8));
    }
}
