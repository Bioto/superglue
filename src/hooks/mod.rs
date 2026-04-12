//! Hook system — observe and optionally mutate the LLM pipeline.
//!
//! Hooks fire at well-defined points in `complete_with_tools`, `stream_complete`,
//! and `batch_complete`. Each hook stage carries a [`HookContext`] with a string
//! `content` field; handlers that return a new `content` string can mutate the
//! pipeline for the two mutable stages ([`HookStage::PreTool`] and
//! [`HookStage::PostTool`]). All other stages are observation-only — the return
//! value is used for chaining within the handler list but discarded by the caller.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use superglue::hooks::{HookConfig, HookContext, HookErrorStrategy, HookRegistry, HookStage};
//! use superglue::hooks::HookHandler;
//! use async_trait::async_trait;
//!
//! struct Counter(std::sync::Arc<std::sync::atomic::AtomicU32>);
//!
//! #[async_trait]
//! impl HookHandler for Counter {
//!     async fn execute(&self, ctx: HookContext) -> Result<HookContext, superglue::hooks::HookError> {
//!         self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
//!         Ok(ctx)
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::warn;

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Points in the pipeline at which hooks fire.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookStage {
    /// Before each HTTP call to the LLM (`complete_with_tools`, `stream_complete`).
    /// Content: last user message. **Observation only.**
    PreCompletion,
    /// After each LLM response, before tool dispatch.
    /// Content: assistant text (empty string if model returned `null`). **Observation only.**
    PostCompletion,
    /// Before a tool is invoked. Content: JSON-serialized arguments. **Mutating.**
    PreTool,
    /// After a tool returns. Content: JSON-serialized result. **Mutating.**
    PostTool,
    /// When an HTTP retry is attempted. Content: error description. **Observation only.**
    OnRetry,
    /// Before each item in `batch_complete`. Content: prompt string. **Observation only.**
    PreBatchItem,
    /// After each item in `batch_complete` finishes. Content: response text or error. **Observation only.**
    PostBatchItem,
}

impl HookStage {
    /// Parse from a lowercase snake_case string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pre_completion" => Some(Self::PreCompletion),
            "post_completion" => Some(Self::PostCompletion),
            "pre_tool" => Some(Self::PreTool),
            "post_tool" => Some(Self::PostTool),
            "on_retry" => Some(Self::OnRetry),
            "pre_batch_item" => Some(Self::PreBatchItem),
            "post_batch_item" => Some(Self::PostBatchItem),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Payload passed to every hook handler.
#[derive(Debug, Clone)]
pub struct HookContext {
    /// Current pipeline stage.
    pub stage: HookStage,
    /// Stage-specific string payload (see [`HookStage`] docs).
    pub content: String,
    /// Stage-specific key-value metadata (e.g. `"tool_name"` for tool stages).
    pub metadata: HashMap<String, Value>,
}

impl HookContext {
    /// Construct a context with no metadata.
    pub fn new(stage: HookStage, content: impl Into<String>) -> Self {
        HookContext {
            stage,
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Construct a context with a single metadata entry.
    pub fn with_meta(
        stage: HookStage,
        content: impl Into<String>,
        key: &str,
        value: impl Into<Value>,
    ) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert(key.to_string(), value.into());
        HookContext {
            stage,
            content: content.into(),
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Error strategy
// ---------------------------------------------------------------------------

/// How to handle a hook handler that returns an error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum HookErrorStrategy {
    /// Log the error as a warning and continue with the current (unmodified) content. (default)
    #[default]
    Skip,
    /// Propagate the error, aborting the pipeline call.
    Abort,
}

impl HookErrorStrategy {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "skip" => Some(Self::Skip),
            "abort" => Some(Self::Abort),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by hook execution.
#[derive(Debug)]
pub struct HookError {
    pub hook_name: String,
    pub message: String,
}

impl HookError {
    pub fn new(hook_name: impl Into<String>, message: impl Into<String>) -> Self {
        HookError {
            hook_name: hook_name.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hook '{}' error: {}", self.hook_name, self.message)
    }
}

impl std::error::Error for HookError {}

// ---------------------------------------------------------------------------
// Handler trait
// ---------------------------------------------------------------------------

/// Async hook handler. Implement this to observe or mutate pipeline data.
///
/// The handler receives the current [`HookContext`] and returns a (possibly
/// modified) `HookContext`. For observation-only stages, return `Ok(ctx)`.
/// For mutating stages ([`HookStage::PreTool`], [`HookStage::PostTool`]),
/// modify `ctx.content` and return `Ok(ctx)`.
#[async_trait]
pub trait HookHandler: Send + Sync {
    async fn execute(&self, ctx: HookContext) -> Result<HookContext, HookError>;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Wraps a handler with metadata and an error strategy.
pub struct HookConfig {
    /// Human-readable name used in warning/error messages.
    pub name: String,
    /// What to do when the handler returns an error.
    pub error_strategy: HookErrorStrategy,
    /// The handler implementation.
    pub handler: Arc<dyn HookHandler>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Stores hooks by stage and executes them in registration order.
///
/// An empty registry (the default) adds no overhead — `run()` returns
/// immediately when no hooks are registered for a stage.
#[derive(Default)]
pub struct HookRegistry {
    hooks: RwLock<HashMap<HookStage, Vec<HookConfig>>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handler for the given stage.
    pub async fn add(&self, stage: HookStage, config: HookConfig) {
        let mut map = self.hooks.write().await;
        map.entry(stage).or_default().push(config);
    }

    /// Execute all handlers registered for `stage` in order, chaining the
    /// context through them. Returns the final context.
    ///
    /// Per-handler errors are handled according to [`HookErrorStrategy`]:
    /// - `Skip`: log a warning and keep the context unchanged.
    /// - `Abort`: return `Err(HookError)` immediately.
    pub async fn run(
        &self,
        stage: HookStage,
        mut ctx: HookContext,
    ) -> Result<HookContext, HookError> {
        let map = self.hooks.read().await;
        let configs = match map.get(&stage) {
            Some(v) if !v.is_empty() => v,
            _ => return Ok(ctx),
        };

        for cfg in configs {
            match cfg.handler.execute(ctx.clone()).await {
                Ok(new_ctx) => ctx = new_ctx,
                Err(e) => match cfg.error_strategy {
                    HookErrorStrategy::Skip => {
                        warn!(hook = %cfg.name, error = %e, "hook failed (skip)");
                    }
                    HookErrorStrategy::Abort => return Err(e),
                },
            }
        }
        Ok(ctx)
    }

    /// Returns `true` when no hooks are registered for `stage`.
    pub async fn is_empty_for(&self, stage: &HookStage) -> bool {
        let map = self.hooks.read().await;
        map.get(stage).map_or(true, |v| v.is_empty())
    }
}
