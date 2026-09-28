//! Guardrail system — validate and optionally transform LLM inputs and outputs.
//!
//! Guardrails run at two points in the pipeline:
//!
//! - **Input** — on the user message, before any HTTP call to the LLM.
//!   A block propagates immediately as [`ChatError::Guardrail`].
//! - **Output** — on the final assistant text, after the tool loop.
//!   A block triggers an LLM retry loop (up to [`GuardrailRegistry::max_output_retries`])
//!   where the rejected response is appended with a "please revise" message; only after
//!   all retries are exhausted does the error propagate.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use superglue::guardrails::{
//!     BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailRegistry,
//! };
//! use std::sync::Arc;
//!
//! async fn example() {
//!     let guardrails = GuardrailRegistry::new();
//!
//!     // Block any input containing "badword".
//!     guardrails
//!         .add_input(GuardrailConfig {
//!             name: "blocklist".into(),
//!             handler: Arc::new(
//!                 BlocklistGuardrail::new(&["badword"], BlocklistAction::Block).unwrap(),
//!             ),
//!         })
//!         .await;
//! }
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use tokio::sync::RwLock;
use tracing::warn;

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Which side of the pipeline the guardrail applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GuardrailStage {
    Input,
    Output,
}

impl GuardrailStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            GuardrailStage::Input => "input",
            GuardrailStage::Output => "output",
        }
    }
}

impl std::fmt::Display for GuardrailStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// Result of running a single guardrail handler.
#[derive(Debug)]
pub enum GuardrailOutcome {
    /// Content is allowed (possibly transformed — the inner `String` is the new content).
    Allow(String),
    /// Content is blocked. The inner `String` is the human-readable reason.
    Block(String),
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Returned when content is blocked after all retries are exhausted.
#[derive(Debug)]
pub struct GuardrailError {
    pub stage: GuardrailStage,
    pub reason: String,
    pub guardrail_name: String,
}

impl GuardrailError {
    pub fn new(
        stage: GuardrailStage,
        guardrail_name: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        GuardrailError {
            stage,
            guardrail_name: guardrail_name.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "guardrail '{}' blocked {} content: {}",
            self.guardrail_name, self.stage, self.reason
        )
    }
}

impl std::error::Error for GuardrailError {}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Async guardrail handler.
///
/// Implement this trait to validate or transform LLM inputs/outputs.
/// Return [`GuardrailOutcome::Allow`] with (possibly modified) content to pass
/// through, or [`GuardrailOutcome::Block`] to reject.
#[async_trait]
pub trait GuardrailHandler: Send + Sync {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Wraps a handler with a name for logging/error messages.
pub struct GuardrailConfig {
    pub name: String,
    pub handler: Arc<dyn GuardrailHandler>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Stores input and output guardrails and runs them in registration order.
///
/// An empty registry (the default) adds no overhead.
///
/// `max_output_retries` controls how many times `complete_with_tools` will
/// re-invoke the LLM when the output is rejected (default 3).
pub struct GuardrailRegistry {
    input: RwLock<Vec<GuardrailConfig>>,
    output: RwLock<Vec<GuardrailConfig>>,
    pub max_output_retries: u32,
}

impl Default for GuardrailRegistry {
    fn default() -> Self {
        GuardrailRegistry {
            input: RwLock::new(Vec::new()),
            output: RwLock::new(Vec::new()),
            max_output_retries: 3,
        }
    }
}

impl GuardrailRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_output_retries(mut self, n: u32) -> Self {
        self.max_output_retries = n;
        self
    }

    /// Register a handler that applies to **input** (user messages).
    pub async fn add_input(&self, config: GuardrailConfig) {
        self.input.write().await.push(config);
    }

    /// Register a handler that applies to **output** (assistant responses).
    pub async fn add_output(&self, config: GuardrailConfig) {
        self.output.write().await.push(config);
    }

    /// Run all input handlers in registration order, chaining content transforms.
    /// Returns the first `Block` encountered, or `Allow` with the final content.
    pub async fn run_input(&self, content: &str) -> (GuardrailOutcome, String) {
        self.run_chain(&self.input.read().await, GuardrailStage::Input, content)
            .await
    }

    /// Run all output handlers in registration order, chaining content transforms.
    pub async fn run_output(&self, content: &str) -> (GuardrailOutcome, String) {
        self.run_chain(&self.output.read().await, GuardrailStage::Output, content)
            .await
    }

    /// Returns `true` when no input guardrails are registered.
    pub async fn input_is_empty(&self) -> bool {
        self.input.read().await.is_empty()
    }

    /// Returns `true` when no output guardrails are registered.
    pub async fn output_is_empty(&self) -> bool {
        self.output.read().await.is_empty()
    }

    async fn run_chain(
        &self,
        configs: &[GuardrailConfig],
        stage: GuardrailStage,
        initial: &str,
    ) -> (GuardrailOutcome, String) {
        if configs.is_empty() {
            return (GuardrailOutcome::Allow(initial.to_string()), String::new());
        }
        let guard = crate::telemetry::openinference::GuardSpan::begin(stage.as_str(), initial);
        let mut current = initial.to_string();
        for cfg in configs {
            match cfg.handler.check(stage.clone(), &current).await {
                GuardrailOutcome::Allow(transformed) => {
                    current = transformed;
                }
                GuardrailOutcome::Block(reason) => {
                    guard.finish(&reason, true);
                    return (GuardrailOutcome::Block(reason), cfg.name.clone());
                }
            }
        }
        guard.finish(&current, false);
        (GuardrailOutcome::Allow(current), String::new())
    }
}

// ---------------------------------------------------------------------------
// Built-in: BlocklistGuardrail
// ---------------------------------------------------------------------------

/// Action taken when a blocklist pattern matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlocklistAction {
    /// Reject the content entirely — returns `GuardrailOutcome::Block`.
    Block,
    /// Replace the matching text with `[REDACTED]` — returns `GuardrailOutcome::Allow`.
    Redact,
}

/// Guardrail that matches content against a list of regex patterns.
///
/// - `Block`: any match causes the content to be rejected.
/// - `Redact`: all matches are replaced with `[REDACTED]` and the content is allowed through.
pub struct BlocklistGuardrail {
    patterns: Vec<Regex>,
    action: BlocklistAction,
    stages: HashSet<GuardrailStage>,
}

impl BlocklistGuardrail {
    /// Create a new blocklist guardrail.
    ///
    /// `patterns` is a list of regex strings. Returns an error if any pattern is invalid.
    /// `stages` controls which pipeline stages this guardrail applies to. Pass
    /// `[GuardrailStage::Input, GuardrailStage::Output]` to apply to both, or a single
    /// stage for one direction only.
    pub fn new(
        patterns: &[impl AsRef<str>],
        action: BlocklistAction,
    ) -> Result<Self, regex::Error> {
        let compiled = patterns
            .iter()
            .map(|p| Regex::new(p.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut stages = HashSet::new();
        stages.insert(GuardrailStage::Input);
        stages.insert(GuardrailStage::Output);
        Ok(BlocklistGuardrail {
            patterns: compiled,
            action,
            stages,
        })
    }

    /// Restrict this guardrail to only input or only output.
    pub fn for_stages(mut self, stages: impl IntoIterator<Item = GuardrailStage>) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }
}

#[async_trait]
impl GuardrailHandler for BlocklistGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        if !self.stages.contains(&stage) {
            return GuardrailOutcome::Allow(content.to_string());
        }
        match self.action {
            BlocklistAction::Block => {
                for pat in &self.patterns {
                    if pat.is_match(content) {
                        return GuardrailOutcome::Block(format!(
                            "matched blocklist pattern /{}/",
                            pat.as_str()
                        ));
                    }
                }
                GuardrailOutcome::Allow(content.to_string())
            }
            BlocklistAction::Redact => {
                let mut result = content.to_string();
                for pat in &self.patterns {
                    result = pat.replace_all(&result, "[REDACTED]").into_owned();
                }
                GuardrailOutcome::Allow(result)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in: MaxLengthGuardrail
// ---------------------------------------------------------------------------

/// Strategy when content exceeds the length limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengthStrategy {
    /// Silently truncate the content to the limit (character-level).
    Truncate,
    /// Reject the content — returns `GuardrailOutcome::Block`.
    Block,
}

/// Guardrail that enforces maximum character lengths on inputs and/or outputs.
pub struct MaxLengthGuardrail {
    max_input_chars: Option<usize>,
    max_output_chars: Option<usize>,
    strategy: LengthStrategy,
}

impl MaxLengthGuardrail {
    pub fn new(
        max_input_chars: Option<usize>,
        max_output_chars: Option<usize>,
        strategy: LengthStrategy,
    ) -> Self {
        MaxLengthGuardrail {
            max_input_chars,
            max_output_chars,
            strategy,
        }
    }
}

#[async_trait]
impl GuardrailHandler for MaxLengthGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let limit = match stage {
            GuardrailStage::Input => self.max_input_chars,
            GuardrailStage::Output => self.max_output_chars,
        };
        let Some(max) = limit else {
            return GuardrailOutcome::Allow(content.to_string());
        };
        if content.len() <= max {
            return GuardrailOutcome::Allow(content.to_string());
        }
        match self.strategy {
            LengthStrategy::Truncate => {
                let truncated = content.chars().take(max).collect();
                GuardrailOutcome::Allow(truncated)
            }
            LengthStrategy::Block => GuardrailOutcome::Block(format!(
                "content length {} exceeds limit of {} chars",
                content.len(),
                max
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in: PiiRedactGuardrail
// ---------------------------------------------------------------------------

/// Guardrail that automatically redacts common PII patterns (emails, US phone
/// numbers, SSNs, credit card numbers) by replacing them with `[REDACTED]`.
///
/// This is always an allow-with-transform guardrail — it never blocks.
pub struct PiiRedactGuardrail {
    patterns: Vec<Regex>,
    stages: HashSet<GuardrailStage>,
}

impl PiiRedactGuardrail {
    /// Build a PII redaction guardrail with the default pattern set.
    pub fn new() -> Self {
        let raw = [
            // email
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            // US phone (various formats)
            r"\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
            // SSN
            r"\b\d{3}-\d{2}-\d{4}\b",
            // 16-digit credit card
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        ];
        let patterns = raw
            .iter()
            .filter_map(|p| {
                Regex::new(p)
                    .map_err(|e| warn!("PII pattern compile error: {e}"))
                    .ok()
            })
            .collect();
        let mut stages = HashSet::new();
        stages.insert(GuardrailStage::Input);
        stages.insert(GuardrailStage::Output);
        PiiRedactGuardrail { patterns, stages }
    }

    pub fn for_stages(mut self, stages: impl IntoIterator<Item = GuardrailStage>) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }
}

impl Default for PiiRedactGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GuardrailHandler for PiiRedactGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        if !self.stages.contains(&stage) {
            return GuardrailOutcome::Allow(content.to_string());
        }
        let mut result = content.to_string();
        for pat in &self.patterns {
            result = pat.replace_all(&result, "[REDACTED]").into_owned();
        }
        GuardrailOutcome::Allow(result)
    }
}

// ---------------------------------------------------------------------------
// Built-in: SecretRedactGuardrail
// ---------------------------------------------------------------------------

/// Guardrail that redacts common secret/credential patterns in text.
///
/// Always allow-with-transform — never blocks.
pub struct SecretRedactGuardrail {
    patterns: Vec<Regex>,
    stages: HashSet<GuardrailStage>,
}

impl SecretRedactGuardrail {
    pub fn new() -> Self {
        let raw = [
            // OpenAI / generic API keys
            r"\bsk-[a-zA-Z0-9]{20,}\b",
            r"\bsk-proj-[a-zA-Z0-9_-]{20,}\b",
            // AWS
            r"\bAKIA[0-9A-Z]{16}\b",
            // GitHub
            r"\bghp_[a-zA-Z0-9]{36,}\b",
            r"\bgithub_pat_[a-zA-Z0-9_]{20,}\b",
            // Slack
            r"\bxox[baprs]-[a-zA-Z0-9-]{10,}\b",
            // Bearer tokens
            r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b",
            // PEM private keys
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            // api_key= style assignments
            r#"(?i)(api[_-]?key|secret|token|password)\s*=\s*[^\s&"']+"#,
        ];
        let patterns = raw
            .iter()
            .filter_map(|p| {
                Regex::new(p)
                    .map_err(|e| warn!("secret pattern compile error: {e}"))
                    .ok()
            })
            .collect();
        let mut stages = HashSet::new();
        stages.insert(GuardrailStage::Input);
        stages.insert(GuardrailStage::Output);
        SecretRedactGuardrail { patterns, stages }
    }

    pub fn for_stages(mut self, stages: impl IntoIterator<Item = GuardrailStage>) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }
}

impl Default for SecretRedactGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GuardrailHandler for SecretRedactGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        if !self.stages.contains(&stage) {
            return GuardrailOutcome::Allow(content.to_string());
        }
        let mut result = content.to_string();
        for pat in &self.patterns {
            result = pat.replace_all(&result, "[REDACTED]").into_owned();
        }
        GuardrailOutcome::Allow(result)
    }
}

#[cfg(test)]
mod secret_redact_tests {
    use super::*;

    #[tokio::test]
    async fn redacts_openai_key() {
        let g = SecretRedactGuardrail::new();
        let (outcome, _) = match g
            .check(
                GuardrailStage::Output,
                "key sk-abcdefghijklmnopqrstuvwxyz123456",
            )
            .await
        {
            GuardrailOutcome::Allow(s) => (s, ()),
            GuardrailOutcome::Block(r) => panic!("unexpected block: {r}"),
        };
        assert!(outcome.contains("[REDACTED]"));
        assert!(!outcome.contains("sk-abc"));
    }

    #[tokio::test]
    async fn leaves_prose_intact() {
        let g = SecretRedactGuardrail::new();
        let text = "Use cargo test to run the suite.";
        match g.check(GuardrailStage::Output, text).await {
            GuardrailOutcome::Allow(s) => assert_eq!(s, text),
            GuardrailOutcome::Block(r) => panic!("unexpected block: {r}"),
        }
    }
}
