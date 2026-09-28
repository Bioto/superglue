//! Static model pricing and cost estimation helpers.

mod generation;

use serde_json::Value;

use crate::proto;

pub use generation::fetch_generation_cost_usd;

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
    if m.contains("deepseek") {
        return Some(ModelRates {
            prompt_per_m: 0.15,
            completion_per_m: 1.20,
            cached_prompt_multiplier: 0.02,
        });
    }
    None
}

/// Accept a finite billed USD amount, including `0.0`.
#[must_use]
pub fn sanitize_billed_cost_usd(cost: Option<f64>) -> Option<f64> {
    cost.filter(|c| c.is_finite() && *c >= 0.0)
}

pub(crate) fn json_cost_number(value: &Value) -> Option<f64> {
    let cost = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        _ => return None,
    };
    sanitize_billed_cost_usd(Some(cost))
}

/// Read billed USD from a provider usage object.
///
/// Prefers `cost`, then `cost_usd`, then `total_cost`. Does not read
/// `cost_details.upstream_inference_cost`.
#[must_use]
pub fn billed_cost_usd_from_usage_json(usage: &Value) -> Option<f64> {
    for key in ["cost", "cost_usd", "total_cost"] {
        if let Some(value) = usage.get(key)
            && let Some(cost) = json_cost_number(value)
        {
            return Some(cost);
        }
    }
    None
}

/// Prefer a billed amount when present; otherwise estimate from the static table.
#[must_use]
pub fn resolve_model_call_cost_usd(model: &str, usage: &proto::Usage, billed: Option<f64>) -> f64 {
    if let Some(cost) = sanitize_billed_cost_usd(billed) {
        return cost;
    }
    if let Some(cost) = sanitize_billed_cost_usd(usage.cost_usd) {
        return cost;
    }
    estimate_model_call_cost_usd(model, usage)
}

/// Write the resolved USD amount onto `usage.cost_usd` and return it.
pub fn apply_resolved_cost_usd(model: &str, usage: &mut proto::Usage, billed: Option<f64>) -> f64 {
    let cost = resolve_model_call_cost_usd(model, usage, billed);
    usage.cost_usd = Some(cost);
    cost
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

    fn test_usage(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: Option<u32>,
    ) -> proto::Usage {
        proto::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            cached_tokens,
            reasoning_tokens: None,
            cost_usd: None,
        }
    }

    #[test]
    fn estimates_nano_pricing() {
        let usage = test_usage(1_000_000, 0, None);
        let cost = estimate_model_call_cost_usd("gpt-5.4-nano-2026-03-17-mini", &usage);
        assert!((cost - 0.15).abs() < 1e-9);
    }

    #[test]
    fn applies_cached_prompt_discount() {
        let usage = test_usage(1_000_000, 0, Some(800_000));
        let cost = estimate_model_call_cost_usd("gpt-5.4-nano-2026-03-17-mini", &usage);
        let expected = (200_000.0 * 0.15 + 800_000.0 * 0.15 * 0.5) / 1_000_000.0;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_is_zero() {
        let usage = test_usage(100, 100, None);
        assert_eq!(estimate_model_call_cost_usd("unknown-model", &usage), 0.0);
    }

    #[test]
    fn prefers_billed_cost_over_table() {
        let usage = test_usage(1_000_000, 0, None);
        let cost = resolve_model_call_cost_usd("gpt-4o-mini", &usage, Some(0.0123));
        assert!((cost - 0.0123).abs() < 1e-9);
    }

    #[test]
    fn prefers_billed_zero_over_table() {
        let usage = test_usage(1_000_000, 0, None);
        let cost = resolve_model_call_cost_usd("gpt-4o-mini", &usage, Some(0.0));
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn missing_or_invalid_billed_falls_back_to_estimate() {
        let usage = test_usage(1_000_000, 0, None);
        let estimated = estimate_model_call_cost_usd("gpt-4o-mini", &usage);
        assert!(estimated > 0.0);
        assert_eq!(
            resolve_model_call_cost_usd("gpt-4o-mini", &usage, None),
            estimated
        );
        assert_eq!(
            resolve_model_call_cost_usd("gpt-4o-mini", &usage, Some(f64::NAN)),
            estimated
        );
        assert_eq!(
            resolve_model_call_cost_usd("gpt-4o-mini", &usage, Some(-0.01)),
            estimated
        );
    }

    #[test]
    fn deepseek_without_billed_cost_is_nonzero() {
        let usage = test_usage(1_000_000, 1_000_000, None);
        let cost =
            resolve_model_call_cost_usd("openrouter:deepseek/deepseek-v4.1-flash", &usage, None);
        assert!((cost - 1.35).abs() < 1e-9);
    }

    #[test]
    fn reads_cost_aliases_from_usage_json() {
        assert_eq!(
            billed_cost_usd_from_usage_json(&serde_json::json!({"cost": 0.01})),
            Some(0.01)
        );
        assert_eq!(
            billed_cost_usd_from_usage_json(&serde_json::json!({"cost_usd": 0.02})),
            Some(0.02)
        );
        assert_eq!(
            billed_cost_usd_from_usage_json(&serde_json::json!({"total_cost": 0.03})),
            Some(0.03)
        );
        assert_eq!(
            billed_cost_usd_from_usage_json(&serde_json::json!({
                "cost": -1.0,
                "total_cost": 0.03
            })),
            Some(0.03)
        );
        assert_eq!(
            billed_cost_usd_from_usage_json(&serde_json::json!({
                "cost_details": { "upstream_inference_cost": 9.0 }
            })),
            None
        );
    }
}
