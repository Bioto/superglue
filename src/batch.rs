//! Batch chat completions: run many prompts concurrently with a shared HTTP
//! client and tool registry.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use superglue::batch::{BatchConfig, BatchRequest, batch_complete};
//! use superglue::chat::ChatOptions;
//! use superglue::guardrails::GuardrailRegistry;
//! use superglue::hooks::HookRegistry;
//! use superglue::http::{ClientConfig, HttpClient};
//! use superglue::tools::ToolRegistry;
//!
//! async fn example() {
//!     let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
//!     let registry = Arc::new(ToolRegistry::new());
//!     let hooks = Arc::new(HookRegistry::new());
//!     let guardrails = Arc::new(GuardrailRegistry::new());
//!     let options = ChatOptions::new("https://api.openai.com", "sk-...", "gpt-5.4-nano-2026-03-17-mini");
//!
//!     let requests = vec![
//!         BatchRequest::new("What is the capital of France?"),
//!         BatchRequest::new("What is 2 + 2?"),
//!     ];
//!
//!     let response = batch_complete(http, registry, hooks, guardrails, requests, &options, BatchConfig::default())
//!         .await
//!         .unwrap();
//!
//!     println!("{}/{} succeeded", response.successful, response.total_requests);
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::info;

use std::time::Duration;

use crate::cancel::CancellationToken;
use crate::chat::{ChatOptions, complete_with_tools};
use crate::guardrails::GuardrailRegistry;
use crate::hooks::{HookContext, HookRegistry, HookStage};
use crate::http::HttpClient;
use crate::openai::ChatMessage;
use crate::proto;
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// Error strategy
// ---------------------------------------------------------------------------

/// How the batch handles per-request failures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ErrorStrategy {
    /// Collect all results, including failures. Failed items have `success = false`. (default)
    #[default]
    Continue,
    /// Return only successful results; silently drop failures.
    Skip,
    /// Abort as soon as any request fails and return an error.
    FailFast,
}

impl ErrorStrategy {
    /// Parse from a case-insensitive string (`"continue"`, `"skip"`, `"fail_fast"`).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "continue" => Some(Self::Continue),
            "skip" => Some(Self::Skip),
            "fail_fast" | "failfast" => Some(Self::FailFast),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Request / config
// ---------------------------------------------------------------------------

/// A single item in a batch.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Optional caller-supplied identifier. Auto-generated if `None`.
    pub id: Option<String>,
    /// The user message to send.
    pub prompt: String,
    /// Per-request system prompt. Overrides [`ChatOptions::system_prompt`] when set.
    pub system_prompt: Option<String>,
    /// Arbitrary key-value data passed through to [`BatchResult::metadata`].
    pub metadata: HashMap<String, serde_json::Value>,
}

impl BatchRequest {
    /// Create a minimal request with just a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        BatchRequest {
            id: None,
            prompt: prompt.into(),
            system_prompt: None,
            metadata: HashMap::new(),
        }
    }

    /// Assign a caller-supplied ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Override the system prompt for this request only.
    pub fn with_system_prompt(mut self, sp: impl Into<String>) -> Self {
        self.system_prompt = Some(sp.into());
        self
    }
}

/// Configuration for [`batch_complete`].
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of in-flight completion calls at once (default 5).
    pub max_concurrent: usize,
    /// How to handle per-request failures (default [`ErrorStrategy::Continue`]).
    pub error_strategy: ErrorStrategy,
    /// Per-request total timeout (connect + response body). Overrides the
    /// `HttpClient`'s configured timeout when set.
    pub timeout: Option<Duration>,
    /// Per-request connect timeout. Overrides the `HttpClient`'s configured
    /// connect timeout when set.
    pub connect_timeout: Option<Duration>,
    /// Optional cancellation token. When cancelled, all in-flight and pending
    /// batch items are aborted and [`BatchError::Cancelled`] is returned.
    pub cancel: Option<CancellationToken>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            max_concurrent: 5,
            error_strategy: ErrorStrategy::Continue,
            timeout: None,
            connect_timeout: None,
            cancel: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome of a single item in a batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Identifier for this request (auto-generated if not supplied).
    pub id: String,
    /// Whether the completion succeeded.
    pub success: bool,
    /// Assistant response text (set on success).
    pub content: Option<String>,
    /// Error message (set on failure).
    pub error: Option<String>,
    /// Number of completion HTTP calls made for this item.
    pub rounds: u32,
    /// Token usage from the final completion call.
    pub usage: Option<proto::Usage>,
    /// Wall-clock seconds for this item (including retries).
    pub elapsed_secs: f64,
    /// Caller-supplied metadata passed through from [`BatchRequest::metadata`].
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Aggregate outcome of a [`batch_complete`] call.
#[derive(Debug, Clone)]
pub struct BatchResponse {
    /// Per-item results (order matches input unless [`ErrorStrategy::Skip`] is used).
    pub results: Vec<BatchResult>,
    /// Total number of input requests (before any filtering).
    pub total_requests: usize,
    /// Number of items with `success = true`.
    pub successful: usize,
    /// Number of items with `success = false`.
    pub failed: usize,
    /// Total wall-clock seconds for the whole batch.
    pub elapsed_secs: f64,
    /// Sum of token usage across all successful completions.
    pub total_usage: Option<proto::Usage>,
}

// ---------------------------------------------------------------------------
// Batch error
// ---------------------------------------------------------------------------

/// Errors returned by [`batch_complete`].
#[derive(Debug)]
pub enum BatchError {
    /// A request failed and [`ErrorStrategy::FailFast`] was active.
    FailFast { id: String, message: String },
    /// The concurrency infrastructure failed (should never happen in practice).
    Internal(String),
    /// The batch was cancelled via the [`BatchConfig::cancel`] token.
    Cancelled,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::FailFast { id, message } => {
                write!(f, "batch aborted on request '{id}': {message}")
            }
            BatchError::Internal(msg) => write!(f, "batch internal error: {msg}"),
            BatchError::Cancelled => write!(f, "batch cancelled"),
        }
    }
}

impl std::error::Error for BatchError {}

// ---------------------------------------------------------------------------
// Internal task result
// ---------------------------------------------------------------------------

/// Carries the original error string alongside the task index for FailFast.
struct TaskOutcome {
    idx: usize,
    result: BatchResult,
}

// ---------------------------------------------------------------------------
// Core function
// ---------------------------------------------------------------------------

/// Run many prompts concurrently, sharing a single `HttpClient` and `ToolRegistry`.
///
/// At most `config.max_concurrent` requests are in flight at any time.
/// The `options` passed here act as defaults; per-request `system_prompt` overrides
/// [`ChatOptions::system_prompt`] when set.
///
/// # Errors
///
/// Only returns `Err` when [`ErrorStrategy::FailFast`] is configured and a
/// request fails. Otherwise errors are captured per-item in [`BatchResult`].
pub async fn batch_complete(
    http: Arc<HttpClient>,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    mut requests: Vec<BatchRequest>,
    options: &ChatOptions,
    config: BatchConfig,
) -> Result<BatchResponse, BatchError> {
    let batch_start = Instant::now();
    let total_requests = requests.len();

    if total_requests == 0 {
        return Ok(BatchResponse {
            results: vec![],
            total_requests: 0,
            successful: 0,
            failed: 0,
            elapsed_secs: 0.0,
            total_usage: None,
        });
    }

    // Assign IDs to any request that doesn't already have one.
    for (i, req) in requests.iter_mut().enumerate() {
        if req.id.is_none() {
            req.id = Some(format!("batch-{i}"));
        }
    }

    // If per-batch timeout overrides are set, derive a new HttpClient for this batch.
    // The derived client inherits retry policy and QPS limiter from the parent.
    let http: Arc<HttpClient> = if config.timeout.is_some() || config.connect_timeout.is_some() {
        let t = config.timeout.unwrap_or(http.config.timeout);
        let ct = config
            .connect_timeout
            .unwrap_or(http.config.connect_timeout);
        Arc::new(
            http.clone_with_timeouts(t, ct)
                .map_err(|e| BatchError::Internal(e.to_string()))?,
        )
    } else {
        Arc::clone(&http)
    };

    metrics::histogram!(crate::telemetry::metrics::BATCH_SIZE).record(total_requests as f64);

    let sem = Arc::new(Semaphore::new(config.max_concurrent));
    let base_options = Arc::new(options.clone());
    let error_strategy = config.error_strategy.clone();

    info!(
        total_requests,
        model = %options.model,
        error_strategy = ?error_strategy,
        max_concurrent = config.max_concurrent,
        "batch_complete started"
    );
    let cancel_token = config.cancel.clone();

    // Spawn one task per request.
    let mut join_set: JoinSet<TaskOutcome> = JoinSet::new();

    for (idx, request) in requests.into_iter().enumerate() {
        let sem = Arc::clone(&sem);
        let http = Arc::clone(&http);
        let registry = Arc::clone(&registry);
        let hooks = Arc::clone(&hooks);
        let guardrails = Arc::clone(&guardrails);
        let mut opts = (*base_options).clone();

        // Per-request system prompt override.
        if let Some(sp) = request.system_prompt.clone() {
            opts.system_prompt = Some(sp);
        }

        let id = request.id.clone().unwrap_or_else(|| format!("batch-{idx}"));
        let prompt = request.prompt.clone();
        let metadata = request.metadata.clone();

        // Wire the batch item's id as the request_id for correlation.
        opts.request_id = Some(id.clone());
        // Propagate the batch-level cancellation token into each per-item options.
        if let Some(ref token) = cancel_token {
            opts.cancel = Some(token.child_token());
        }

        join_set.spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore closed");
            let item_start = Instant::now();

            // --- PreBatchItem hook (observation only) ---
            let _ = hooks
                .run(
                    HookStage::PreBatchItem,
                    HookContext::with_meta(HookStage::PreBatchItem, &prompt, "id", id.as_str()),
                )
                .await;

            let messages = vec![ChatMessage::text("user", prompt)];
            let outcome =
                complete_with_tools(&http, &registry, &hooks, &guardrails, messages, &opts).await;

            let elapsed_secs = item_start.elapsed().as_secs_f64();

            let result = match outcome {
                Ok(out) => {
                    let content_str = out.content.clone().unwrap_or_default();
                    // --- PostBatchItem hook (observation only) ---
                    let _ = hooks
                        .run(
                            HookStage::PostBatchItem,
                            HookContext::with_meta(
                                HookStage::PostBatchItem,
                                &content_str,
                                "id",
                                id.as_str(),
                            ),
                        )
                        .await;
                    BatchResult {
                        id,
                        success: true,
                        content: out.content,
                        error: None,
                        rounds: out.rounds,
                        usage: out.usage,
                        elapsed_secs,
                        metadata,
                    }
                }
                Err(e) => {
                    let err_str = e.to_string();
                    // --- PostBatchItem hook (observation only, error case) ---
                    let _ = hooks
                        .run(
                            HookStage::PostBatchItem,
                            HookContext::with_meta(
                                HookStage::PostBatchItem,
                                &err_str,
                                "id",
                                id.as_str(),
                            ),
                        )
                        .await;
                    BatchResult {
                        id,
                        success: false,
                        content: None,
                        error: Some(err_str),
                        rounds: 0,
                        usage: None,
                        elapsed_secs,
                        metadata,
                    }
                }
            };

            TaskOutcome { idx, result }
        });
    }

    // Collect results as tasks complete, in whatever order they finish.
    let mut indexed: Vec<(usize, BatchResult)> = Vec::with_capacity(total_requests);

    loop {
        let next = if let Some(ref token) = cancel_token {
            tokio::select! {
                biased;
                _ = token.cancelled() => {
                    join_set.abort_all();
                    return Err(BatchError::Cancelled);
                }
                joined = join_set.join_next() => joined,
            }
        } else {
            join_set.join_next().await
        };

        let Some(joined) = next else { break };

        let TaskOutcome { idx, result } =
            joined.map_err(|e| BatchError::Internal(e.to_string()))?;

        // FailFast: propagate immediately, cancelling remaining tasks implicitly.
        if !result.success && error_strategy == ErrorStrategy::FailFast {
            join_set.abort_all();
            return Err(BatchError::FailFast {
                id: result.id,
                message: result.error.unwrap_or_else(|| "unknown error".into()),
            });
        }

        indexed.push((idx, result));
    }

    // Restore original input order.
    indexed.sort_by_key(|(idx, _)| *idx);
    let mut all_results: Vec<BatchResult> = indexed.into_iter().map(|(_, r)| r).collect();

    // Apply Skip: remove failures from the output.
    if error_strategy == ErrorStrategy::Skip {
        all_results.retain(|r| r.success);
    }

    let successful = all_results.iter().filter(|r| r.success).count();
    let failed = all_results.iter().filter(|r| !r.success).count();

    // Aggregate token usage.
    let total_usage = aggregate_usage(all_results.iter().filter_map(|r| r.usage.as_ref()));

    let elapsed_secs = batch_start.elapsed().as_secs_f64();
    metrics::histogram!(crate::telemetry::metrics::BATCH_DURATION_MS).record(elapsed_secs * 1000.0);

    info!(
        total_requests,
        successful,
        failed,
        elapsed_secs,
        model = %options.model,
        "batch_complete finished"
    );

    Ok(BatchResponse {
        results: all_results,
        total_requests,
        successful,
        failed,
        elapsed_secs,
        total_usage,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn aggregate_usage<'a>(iter: impl Iterator<Item = &'a proto::Usage>) -> Option<proto::Usage> {
    let mut any = false;
    let mut prompt = 0u32;
    let mut completion = 0u32;
    let mut total = 0u32;
    for u in iter {
        any = true;
        prompt += u.prompt_tokens;
        completion += u.completion_tokens;
        total += u.total_tokens;
    }
    if any {
        Some(proto::Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
            cached_tokens: None,
            reasoning_tokens: None,
        })
    } else {
        None
    }
}
