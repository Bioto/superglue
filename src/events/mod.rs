//! Typed process events for LLM observability (StatusEmitter fan-out).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::RwLock;
use tracing::warn;

use crate::proto;

/// Kind of process event emitted during completion / tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessEventKind {
    LlmCallStart,
    LlmCallEnd,
    LlmCallError,
    ToolCallStart,
    ToolCallEnd,
    ToolRoute,
    ReasoningDelta,
}

impl ProcessEventKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LlmCallStart => "llm_call_start",
            Self::LlmCallEnd => "llm_call_end",
            Self::LlmCallError => "llm_call_error",
            Self::ToolCallStart => "tool_call_start",
            Self::ToolCallEnd => "tool_call_end",
            Self::ToolRoute => "tool_route",
            Self::ReasoningDelta => "reasoning_delta",
        }
    }
}

/// A single observability event during an LLM run.
#[derive(Debug, Clone)]
pub struct ProcessEvent {
    pub kind: ProcessEventKind,
    pub request_id: String,
    pub round: u32,
    pub model: String,
    pub tool_call_count: u32,
    pub usage: Option<proto::Usage>,
    pub estimated_cost_usd: Option<f64>,
    pub error_type: Option<String>,
    pub metadata: HashMap<String, String>,
    pub timestamp_ms: i64,
}

impl ProcessEvent {
    #[must_use]
    pub fn new(
        kind: ProcessEventKind,
        request_id: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            request_id: request_id.into(),
            round: 0,
            model: model.into(),
            tool_call_count: 0,
            usage: None,
            estimated_cost_usd: None,
            error_type: None,
            metadata: HashMap::new(),
            timestamp_ms: now_ms(),
        }
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// Receives process events (non-mutating observers).
#[async_trait]
pub trait StatusSubscriber: Send + Sync {
    async fn on_event(&self, event: ProcessEvent);
}

struct StatusEmitterInner {
    subscribers: RwLock<Vec<Arc<dyn StatusSubscriber>>>,
}

/// Fan-out dispatcher for [`ProcessEvent`] subscribers.
#[derive(Clone)]
pub struct StatusEmitter {
    inner: Arc<StatusEmitterInner>,
}

impl std::fmt::Debug for StatusEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatusEmitter").finish_non_exhaustive()
    }
}

impl Default for StatusEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl StatusEmitter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StatusEmitterInner {
                subscribers: RwLock::new(Vec::new()),
            }),
        }
    }

    pub async fn subscribe(&self, subscriber: Arc<dyn StatusSubscriber>) {
        self.inner.subscribers.write().await.push(subscriber);
    }

    pub async fn emit(&self, event: ProcessEvent) {
        let subs = self.inner.subscribers.read().await.clone();
        for sub in subs {
            sub.on_event(event.clone()).await;
        }
    }
}

/// In-memory subscriber that collects events (for tests and audit).
pub struct CollectingSubscriber {
    events: RwLock<Vec<ProcessEvent>>,
}

impl CollectingSubscriber {
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
        }
    }

    pub async fn take_events(&self) -> Vec<ProcessEvent> {
        std::mem::take(&mut *self.events.write().await)
    }
}

impl Default for CollectingSubscriber {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StatusSubscriber for CollectingSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        self.events.write().await.push(event);
    }
}

/// Wraps a sync callback.
pub struct FnSubscriber<F>(pub F);

#[async_trait]
impl<F> StatusSubscriber for FnSubscriber<F>
where
    F: Fn(ProcessEvent) + Send + Sync,
{
    async fn on_event(&self, event: ProcessEvent) {
        (self.0)(event);
    }
}

/// Emit when an emitter is configured.
pub async fn emit_safe(emitter: Option<&Arc<StatusEmitter>>, event: ProcessEvent) {
    if let Some(em) = emitter {
        em.emit(event).await;
    }
}

#[allow(dead_code)]
fn log_subscriber_error(err: &str) {
    warn!(error = %err, "status subscriber error");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fan_out_to_multiple_subscribers() {
        let emitter = Arc::new(StatusEmitter::new());
        let a = Arc::new(CollectingSubscriber::new());
        let b = Arc::new(CollectingSubscriber::new());
        emitter
            .subscribe(Arc::clone(&a) as Arc<dyn StatusSubscriber>)
            .await;
        emitter
            .subscribe(Arc::clone(&b) as Arc<dyn StatusSubscriber>)
            .await;

        emitter
            .emit(ProcessEvent::new(
                ProcessEventKind::LlmCallEnd,
                "req-1",
                "gpt-5.4",
            ))
            .await;

        assert_eq!(a.take_events().await.len(), 1);
        assert_eq!(b.take_events().await.len(), 1);
    }

    #[tokio::test]
    async fn cloned_emitter_shares_subscribers() {
        let emitter = Arc::new(StatusEmitter::new());
        let collector = Arc::new(CollectingSubscriber::new());
        emitter
            .subscribe(Arc::clone(&collector) as Arc<dyn StatusSubscriber>)
            .await;
        let cloned = emitter.clone();
        cloned
            .emit(ProcessEvent::new(
                ProcessEventKind::LlmCallStart,
                "req-2",
                "gpt-5.4",
            ))
            .await;
        assert_eq!(collector.take_events().await.len(), 1);
    }
}
