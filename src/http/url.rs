//! URL helpers for provider base paths.

/// Join a base URL (e.g. `https://api.openai.com`) with a path (`/v1/chat/completions`).
#[must_use]
pub fn join_base_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_paths() {
        assert_eq!(
            join_base_url("https://api.openai.com", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_base_url("https://api.openai.com/", "v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }
}
