//! Conversation history summarization config.

/// When to compress older conversation history.
#[derive(Debug, Clone)]
pub struct SummarizeContextConfig {
    pub enabled: bool,
    /// Cheap message-count pre-filter before measuring context size.
    pub threshold: usize,
    /// Number of recent messages to retain after summarization.
    pub keep_recent: usize,
    /// Approximate character budget for the message history.
    ///
    /// This is intentionally a character budget rather than a tokenizer-specific
    /// token count. Tool rounds can collapse into a single message, so message
    /// count alone is not a useful proxy for context size.
    pub max_chars: usize,
}

impl Default for SummarizeContextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 20,
            keep_recent: 12,
            max_chars: 800_000,
        }
    }
}
