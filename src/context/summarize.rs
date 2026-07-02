//! Conversation history summarization config.

/// When to compress older conversation history (GlueLLM `SummarizeContextConfig` parity).
#[derive(Debug, Clone)]
pub struct SummarizeContextConfig {
    pub enabled: bool,
    pub threshold: usize,
    pub keep_recent: usize,
}

impl Default for SummarizeContextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 20,
            keep_recent: 6,
        }
    }
}
