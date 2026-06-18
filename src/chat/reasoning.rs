//! Model-aware normalization of OpenAI `reasoning_effort` values.

/// Supported reasoning effort levels (OpenAI Responses / reasoning models).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    /// Parse a provider effort string (case-insensitive).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_api_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

const GPT5_EFFORTS: [ReasoningEffort; 6] = [
    ReasoningEffort::None,
    ReasoningEffort::Minimal,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
];

const O_SERIES_EFFORTS: [ReasoningEffort; 3] = [
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
];

fn model_lower(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

fn is_gpt5_family(model: &str) -> bool {
    let m = model_lower(model);
    m.starts_with("gpt-5") || m.contains("gpt-5.")
}

fn is_o_series(model: &str) -> bool {
    let m = model_lower(model);
    m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.contains("-o1")
        || m.contains("-o3")
        || m.contains("-o4")
}

fn supported_efforts(model: &str) -> &'static [ReasoningEffort] {
    if is_gpt5_family(model) {
        &GPT5_EFFORTS
    } else if is_o_series(model) {
        &O_SERIES_EFFORTS
    } else {
        &[]
    }
}

/// Normalize a parsed effort for the given model, downgrading to the nearest
/// lower supported value when the model does not accept the requested level.
#[must_use]
pub fn normalize_reasoning_effort(model: &str, effort: ReasoningEffort) -> Option<String> {
    let supported = supported_efforts(model);
    if supported.is_empty() {
        return None;
    }
    if supported.contains(&effort) {
        return Some(effort.as_api_str().to_string());
    }
    // Downgrade to the highest supported effort not exceeding the request.
    let chosen = supported
        .iter()
        .rev()
        .find(|&&e| e <= effort)
        .copied()
        .unwrap_or(supported[0]);
    Some(chosen.as_api_str().to_string())
}

/// Parse and normalize a string effort for the given model.
#[must_use]
pub fn normalize_reasoning_effort_str(model: &str, effort: &str) -> Option<String> {
    ReasoningEffort::parse(effort).and_then(|e| normalize_reasoning_effort(model, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt5_accepts_full_range() {
        for effort in GPT5_EFFORTS {
            assert_eq!(
                normalize_reasoning_effort("gpt-5.4", effort).as_deref(),
                Some(effort.as_api_str())
            );
        }
    }

    #[test]
    fn o_series_downgrades_none_to_low() {
        assert_eq!(
            normalize_reasoning_effort("o4-mini", ReasoningEffort::None).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn o_series_downgrades_xhigh_to_high() {
        assert_eq!(
            normalize_reasoning_effort("o3-mini", ReasoningEffort::XHigh).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn non_reasoning_model_returns_none() {
        assert!(normalize_reasoning_effort("gpt-4o", ReasoningEffort::High).is_none());
    }
}
