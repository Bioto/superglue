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
    ///
    /// Used only when a measured window fill is not available.
    pub max_chars: usize,
    /// Caller-supplied model context window in tokens. `None` when unknown.
    pub context_window_tokens: Option<u32>,
    /// Percent of the window that must be filled before mid-turn summarize runs.
    /// `0` disables the window rule and falls back to [`Self::max_chars`].
    pub fill_percent: u8,
}

impl Default for SummarizeContextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 20,
            keep_recent: 6,
            max_chars: 800_000,
            context_window_tokens: None,
            fill_percent: 0,
        }
    }
}
