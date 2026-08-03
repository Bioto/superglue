//! Static model pricing and cost estimation helpers.

use crate::proto;

/// USD cost per 1M prompt (input) tokens.
#[derive(Debug, Clone, Copy)]
struct ModelRates {
    prompt_per_m: f64,
    completion_per_m: f64,
    /// Multiplier applied to cached prompt tokens (typically 0.5 = 50% discount).
    cached_prompt_multiplier: f64,
}

fn rates_for_model(model: &str) -> Option<ModelRates> {
    let m = model.trim().to_ascii_lowercase();
    if m.contains("nano") {
        return Some(ModelRates {
            prompt_per_m: 0.15,
            completion_per_m: 0.60,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("gpt-4o-mini") {
        return Some(ModelRates {
            prompt_per_m: 0.15,
            completion_per_m: 0.60,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("o4") {
        return Some(ModelRates {
            prompt_per_m: 2.50,
            completion_per_m: 10.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("o3") {
        return Some(ModelRates {
            prompt_per_m: 2.0,
            completion_per_m: 8.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("o1") {
        return Some(ModelRates {
            prompt_per_m: 15.0,
            completion_per_m: 60.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.starts_with("gpt-5") || m.contains("gpt-5.") {
        return Some(ModelRates {
            prompt_per_m: 2.50,
            completion_per_m: 15.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("gpt-4o") {
        return Some(ModelRates {
            prompt_per_m: 2.50,
            completion_per_m: 10.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    if m.contains("claude") {
        return Some(ModelRates {
            prompt_per_m: 3.0,
            completion_per_m: 15.0,
            cached_prompt_multiplier: 0.1,
        });
    }
    if m.contains("grok") {
        return Some(ModelRates {
            prompt_per_m: 2.0,
            completion_per_m: 10.0,
            cached_prompt_multiplier: 0.5,
        });
    }
    None
}

/// Estimate USD cost for a single model call from token usage without mutating trackers.
#[must_use]
pub fn estimate_model_call_cost_usd(model: &str, usage: &proto::Usage) -> f64 {
    let Some(rates) = rates_for_model(model) else {
        return 0.0;
    };
    let cached = f64::from(usage.cached_tokens.unwrap_or(0));
    let prompt = f64::from(usage.prompt_tokens);
    let uncached_prompt = (prompt - cached).max(0.0);
    let completion = f64::from(usage.completion_tokens);
    (uncached_prompt * rates.prompt_per_m
        + cached * rates.prompt_per_m * rates.cached_prompt_multiplier
        + completion * rates.completion_per_m)
        / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_nano_pricing() {
        let usage = proto::Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            total_tokens: 1_000_000,
            cached_tokens: None,
            reasoning_tokens: None,
        };
        let cost = estimate_model_call_cost_usd("gpt-5.4-nano-2026-03-17-mini", &usage);
        assert!((cost - 0.15).abs() < 1e-9);
    }

    #[test]
    fn applies_cached_prompt_discount() {
        let usage = proto::Usage {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            total_tokens: 1_000_000,
            cached_tokens: Some(800_000),
            reasoning_tokens: None,
        };
        let cost = estimate_model_call_cost_usd("gpt-5.4-nano-2026-03-17-mini", &usage);
        // 200k full + 800k at 50%
        let expected = (200_000.0 * 0.15 + 800_000.0 * 0.15 * 0.5) / 1_000_000.0;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_is_zero() {
        let usage = proto::Usage {
            prompt_tokens: 100,
            completion_tokens: 100,
            total_tokens: 200,
            cached_tokens: None,
            reasoning_tokens: None,
        };
        assert_eq!(estimate_model_call_cost_usd("unknown-model", &usage), 0.0);
    }
}
