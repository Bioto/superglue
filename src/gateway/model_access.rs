//! Per-key model allowlist matching.

/// Returns true when `model` matches any glob pattern in `patterns`.
///
/// Patterns use `*` as a wildcard over the full model string (e.g. `openai:*`, `*`).
#[must_use]
pub fn is_allowed(model: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return false;
    }
    patterns
        .iter()
        .any(|pattern| glob_match(pattern, model))
}

/// Returns true when the caller has unrestricted model access (master key).
#[must_use]
pub fn is_unrestricted(allowed_models: &Option<Vec<String>>) -> bool {
    allowed_models.is_none()
}

fn glob_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return pattern == value;
    }

    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }

    let mut rest = value;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !rest.starts_with(part) {
                return false;
            }
            rest = &rest[part.len()..];
        } else if i == parts.len() - 1 && !pattern.ends_with('*') {
            if !rest.ends_with(part) {
                return false;
            }
        } else {
            let Some(idx) = rest.find(part) else {
                return false;
            };
            rest = &rest[idx + part.len()..];
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        assert!(is_allowed(
            "openai:gpt-4o-mini",
            &["openai:gpt-4o-mini".into()]
        ));
        assert!(!is_allowed(
            "openai:gpt-4o",
            &["openai:gpt-4o-mini".into()]
        ));
    }

    #[test]
    fn provider_wildcard() {
        assert!(is_allowed("openai:gpt-4o-mini", &["openai:*".into()]));
        assert!(!is_allowed("anthropic:claude", &["openai:*".into()]));
    }

    #[test]
    fn star_matches_all() {
        assert!(is_allowed("anything", &["*".into()]));
    }

    #[test]
    fn empty_patterns_deny() {
        assert!(!is_allowed("openai:gpt-4o-mini", &[]));
    }
}
