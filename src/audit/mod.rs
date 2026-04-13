//! In-memory audit trail for superglue runs.
//!
//! [`RunRecorder`] implements [`HookHandler`] and captures every hook event
//! fired during a `complete_with_tools` call, together with the start/finish
//! timestamps and the final [`CompletionOutcome`], into a [`proto::RunRecord`].
//!
//! [`RunStore`] is an in-memory store with an optional maximum capacity.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use superglue::audit::{RunRecorder, RunStore};
//! use superglue::hooks::{HookConfig, HookErrorStrategy, HookHandler, HookRegistry};
//!
//! # #[tokio::main]
//! # async fn main() {
//! let store = Arc::new(RunStore::new(Some(1000)));
//! let recorder = Arc::new(RunRecorder::new(Arc::clone(&store)));
//!
//! let hooks = HookRegistry::new();
//! for stage in RunRecorder::all_stages() {
//!     hooks.add(stage, HookConfig {
//!         name: "audit".to_string(),
//!         error_strategy: HookErrorStrategy::Skip,
//!         handler: Arc::clone(&recorder) as Arc<dyn HookHandler>,
//!     }).await;
//! }
//!
//! for record in store.list().await {
//!     println!("{}: {} events", record.request_id, record.hook_events.len());
//! }
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::hooks::{HookContext, HookError, HookHandler, HookStage};
use crate::proto;

// ---------------------------------------------------------------------------
// RunStore
// ---------------------------------------------------------------------------

/// In-memory store of completed run records.
///
/// Older records are evicted when `max_capacity` is reached (FIFO).
pub struct RunStore {
    records: RwLock<VecDeque<proto::RunRecord>>,
    max_capacity: Option<usize>,
}

impl RunStore {
    /// Create a new store.
    ///
    /// `max_capacity` caps the number of records retained. When the store is
    /// full the oldest record is dropped to make room. Pass `None` for
    /// unbounded storage (be careful in long-running processes).
    pub fn new(max_capacity: Option<usize>) -> Self {
        RunStore {
            records: RwLock::new(VecDeque::new()),
            max_capacity,
        }
    }

    /// Insert a completed run record.
    pub async fn insert(&self, record: proto::RunRecord) {
        let mut deque = self.records.write().await;
        if let Some(cap) = self.max_capacity {
            while deque.len() >= cap {
                deque.pop_front();
            }
        }
        deque.push_back(record);
    }

    /// Retrieve a record by request ID, or `None` if not found.
    pub async fn get(&self, request_id: &str) -> Option<proto::RunRecord> {
        let deque = self.records.read().await;
        deque.iter().find(|r| r.request_id == request_id).cloned()
    }

    /// Return all records in insertion order (oldest first).
    pub async fn list(&self) -> Vec<proto::RunRecord> {
        self.records.read().await.iter().cloned().collect()
    }

    /// Remove all records.
    pub async fn clear(&self) {
        self.records.write().await.clear();
    }

    /// Number of records currently stored.
    pub async fn len(&self) -> usize {
        self.records.read().await.len()
    }

    /// `true` when no records are stored.
    pub async fn is_empty(&self) -> bool {
        self.records.read().await.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RunRecorder
// ---------------------------------------------------------------------------

/// Accumulates [`proto::HookEvent`]s for a run and flushes a [`proto::RunRecord`]
/// to the [`RunStore`] when [`RunRecorder::finish`] is called.
///
/// For the hook integration, register this as a [`HookHandler`] on all relevant
/// stages. The recorder uses the `request_id` in each [`HookContext::metadata`]
/// (key `"request_id"`) to group events by run.
pub struct RunRecorder {
    store: Arc<RunStore>,
    events: RwLock<Vec<proto::HookEvent>>,
}

impl RunRecorder {
    pub fn new(store: Arc<RunStore>) -> Self {
        RunRecorder {
            store,
            events: RwLock::new(Vec::new()),
        }
    }

    /// All [`HookStage`] variants — useful for registering the recorder on every stage.
    pub fn all_stages() -> Vec<HookStage> {
        vec![
            HookStage::PreCompletion,
            HookStage::PostCompletion,
            HookStage::PreTool,
            HookStage::PostTool,
            HookStage::OnRetry,
            HookStage::PreBatchItem,
            HookStage::PostBatchItem,
        ]
    }

    /// Flush accumulated events into a [`proto::RunRecord`] and store it.
    ///
    /// Call once after the run completes (or fails). Pass the final
    /// [`proto::CompletionOutcome`] and the timestamps from the run.
    pub async fn finish(
        &self,
        request_id: String,
        messages: Vec<proto::ChatMessage>,
        outcome: proto::CompletionOutcome,
        started_at_ms: i64,
        finished_at_ms: i64,
    ) {
        let events = {
            let mut guard = self.events.write().await;
            std::mem::take(&mut *guard)
        };
        let record = proto::RunRecord {
            request_id,
            messages,
            hook_events: events,
            outcome: Some(outcome),
            started_at_ms,
            finished_at_ms,
        };
        self.store.insert(record).await;
    }

    /// Return accumulated events without flushing (for inspection mid-run).
    pub async fn peek_events(&self) -> Vec<proto::HookEvent> {
        self.events.read().await.clone()
    }
}

#[async_trait]
impl HookHandler for RunRecorder {
    async fn execute(&self, ctx: HookContext) -> Result<HookContext, HookError> {
        let stage_name = match ctx.stage {
            HookStage::PreCompletion => "pre_completion",
            HookStage::PostCompletion => "post_completion",
            HookStage::PreTool => "pre_tool",
            HookStage::PostTool => "post_tool",
            HookStage::OnRetry => "on_retry",
            HookStage::PreBatchItem => "pre_batch_item",
            HookStage::PostBatchItem => "post_batch_item",
        };

        let request_id = ctx
            .metadata
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let metadata: std::collections::HashMap<String, String> = ctx
            .metadata
            .iter()
            .map(|(k, v)| {
                // For string values, use the raw string to avoid JSON-quoted output.
                let s = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                (k.clone(), s)
            })
            .collect();

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let event = proto::HookEvent {
            stage: stage_name.to_string(),
            content: ctx.content.clone(),
            metadata,
            request_id,
            timestamp_ms,
        };

        self.events.write().await.push(event);
        Ok(ctx)
    }
}
