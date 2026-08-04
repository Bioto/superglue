//! In-memory audit trail for superglue runs and resume helpers.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::RwLock;

use crate::chat::{ChatOptions, CompletionOutcome, complete_with_tools};
use crate::events::{ProcessEvent, ProcessEventKind, StatusSubscriber};
use crate::hooks::{HookContext, HookError, HookHandler, HookStage};
use crate::http::HttpClient;
use crate::openai::ChatMessage;
use crate::proto;
use crate::responses;

// ---------------------------------------------------------------------------
// RunStore
// ---------------------------------------------------------------------------

/// In-memory store of completed run records.
#[derive(Debug)]
pub struct RunStore {
    records: RwLock<VecDeque<proto::RunRecord>>,
    max_capacity: Option<usize>,
}

impl RunStore {
    pub fn new(max_capacity: Option<usize>) -> Self {
        RunStore {
            records: RwLock::new(VecDeque::new()),
            max_capacity,
        }
    }

    pub async fn insert(&self, record: proto::RunRecord) {
        let mut deque = self.records.write().await;
        if let Some(cap) = self.max_capacity {
            while deque.len() >= cap {
                deque.pop_front();
            }
        }
        deque.push_back(record);
    }

    pub async fn get(&self, request_id: &str) -> Option<proto::RunRecord> {
        let deque = self.records.read().await;
        deque.iter().find(|r| r.request_id == request_id).cloned()
    }

    pub async fn list(&self) -> Vec<proto::RunRecord> {
        self.records.read().await.iter().cloned().collect()
    }

    pub async fn clear(&self) {
        self.records.write().await.clear();
    }

    pub async fn len(&self) -> usize {
        self.records.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.records.read().await.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RunRecorder
// ---------------------------------------------------------------------------

/// Optional metadata stored alongside a finished run.
#[derive(Debug, Clone, Default)]
pub struct FinishMetadata {
    pub options_json: Option<String>,
    pub model_used: Option<String>,
    pub previous_response_id: Option<String>,
}

/// Accumulates hook and process events for a run.
pub struct RunRecorder {
    store: Arc<RunStore>,
    max_hook_events: Option<usize>,
    max_process_events: Option<usize>,
    events: RwLock<VecDeque<proto::HookEvent>>,
    process_events: RwLock<VecDeque<proto::ProcessEvent>>,
}

fn push_capped<T>(deque: &mut VecDeque<T>, item: T, cap: Option<usize>) {
    if let Some(max) = cap {
        while deque.len() >= max {
            deque.pop_front();
        }
    }
    deque.push_back(item);
}

impl RunRecorder {
    pub fn new(store: Arc<RunStore>) -> Self {
        Self::with_caps(store, Some(10_000), Some(10_000))
    }

    pub fn with_caps(
        store: Arc<RunStore>,
        max_hook_events: Option<usize>,
        max_process_events: Option<usize>,
    ) -> Self {
        RunRecorder {
            store,
            max_hook_events,
            max_process_events,
            events: RwLock::new(VecDeque::new()),
            process_events: RwLock::new(VecDeque::new()),
        }
    }

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

    pub async fn finish(
        &self,
        request_id: String,
        messages: Vec<proto::ChatMessage>,
        outcome: proto::CompletionOutcome,
        started_at_ms: i64,
        finished_at_ms: i64,
    ) {
        self.finish_with_metadata(
            request_id,
            messages,
            outcome,
            started_at_ms,
            finished_at_ms,
            FinishMetadata::default(),
        )
        .await;
    }

    pub async fn finish_with_metadata(
        &self,
        request_id: String,
        messages: Vec<proto::ChatMessage>,
        outcome: proto::CompletionOutcome,
        started_at_ms: i64,
        finished_at_ms: i64,
        meta: FinishMetadata,
    ) {
        let events = {
            let mut guard = self.events.write().await;
            std::mem::take(&mut *guard).into()
        };
        let process_events = {
            let mut guard = self.process_events.write().await;
            std::mem::take(&mut *guard).into()
        };
        let record = proto::RunRecord {
            request_id,
            messages,
            hook_events: events,
            outcome: Some(outcome),
            started_at_ms,
            finished_at_ms,
            process_events,
            options_json: meta.options_json,
            model_used: meta.model_used,
            previous_response_id: meta.previous_response_id,
        };
        self.store.insert(record).await;
    }

    pub async fn peek_events(&self) -> Vec<proto::HookEvent> {
        self.events.read().await.iter().cloned().collect()
    }

    /// Persist a completed chat turn into the configured [`RunStore`].
    pub async fn record_chat_completion(
        &self,
        outcome: &CompletionOutcome,
        options: &ChatOptions,
        started_at_ms: i64,
    ) {
        let messages: Vec<proto::ChatMessage> =
            outcome.messages.iter().map(chat_message_to_proto).collect();
        self.finish_with_metadata(
            outcome.request_id.clone(),
            messages,
            completion_outcome_to_proto(outcome),
            started_at_ms,
            now_ms(),
            FinishMetadata {
                options_json: Some(options_snapshot(options)),
                model_used: Some(outcome.model_used.clone()),
                previous_response_id: None,
            },
        )
        .await;
    }

    /// Persist a completed Responses turn into the configured [`RunStore`].
    pub async fn record_response_completion(
        &self,
        outcome: &responses::ResponseOutcome,
        options: &ChatOptions,
        started_at_ms: i64,
    ) {
        let messages: Vec<proto::ChatMessage> =
            outcome.messages.iter().map(chat_message_to_proto).collect();
        self.finish_with_metadata(
            outcome.request_id.clone(),
            messages,
            response_outcome_to_proto(outcome),
            started_at_ms,
            now_ms(),
            FinishMetadata {
                options_json: Some(options_snapshot(options)),
                model_used: Some(outcome.model_used.clone()),
                previous_response_id: if outcome.id.is_empty() {
                    None
                } else {
                    Some(outcome.id.clone())
                },
            },
        )
        .await;
    }

    fn record_process_event(&self, event: ProcessEvent) {
        let proto_event = proto::ProcessEvent {
            kind: process_kind_to_proto(event.kind) as i32,
            request_id: event.request_id,
            round: event.round,
            model: event.model,
            tool_call_count: event.tool_call_count,
            usage: event.usage,
            estimated_cost_usd: event.estimated_cost_usd,
            error_type: event.error_type,
            timestamp_ms: event.timestamp_ms,
            metadata: event.metadata,
        };
        if let Ok(mut guard) = self.process_events.try_write() {
            push_capped(&mut guard, proto_event, self.max_process_events);
        }
    }
}

fn process_kind_to_proto(kind: ProcessEventKind) -> proto::ProcessEventKind {
    match kind {
        ProcessEventKind::LlmCallStart => proto::ProcessEventKind::LlmCallStart,
        ProcessEventKind::LlmCallEnd => proto::ProcessEventKind::LlmCallEnd,
        ProcessEventKind::LlmCallError => proto::ProcessEventKind::LlmCallError,
        ProcessEventKind::ToolCallStart => proto::ProcessEventKind::ToolCallStart,
        ProcessEventKind::ToolCallEnd => proto::ProcessEventKind::ToolCallEnd,
        ProcessEventKind::ToolRoute => proto::ProcessEventKind::ToolRoute,
        ProcessEventKind::ReasoningDelta => proto::ProcessEventKind::ReasoningDelta,
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
                let s = v
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string());
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

        let mut guard = self.events.write().await;
        push_capped(&mut guard, event, self.max_hook_events);
        Ok(ctx)
    }
}

#[async_trait]
impl StatusSubscriber for RunRecorder {
    async fn on_event(&self, event: ProcessEvent) {
        self.record_process_event(event);
    }
}

// ---------------------------------------------------------------------------
// Resume helpers
// ---------------------------------------------------------------------------

/// Restored run state for continuing a recorded chat or Responses run.
#[derive(Debug, Clone)]
pub struct ResumeContext {
    pub options: ChatOptions,
    pub messages: Vec<ChatMessage>,
    pub previous_response_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResumeError {
    #[error("run record missing stored options")]
    MissingOptions,
    #[error("failed to parse stored options: {0}")]
    InvalidOptions(String),
}

/// Current Unix timestamp in milliseconds.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[must_use]
pub fn chat_message_to_proto(m: &ChatMessage) -> proto::ChatMessage {
    let tool_calls = m
        .tool_calls
        .as_ref()
        .map(|tcs| {
            tcs.iter()
                .map(|tc| proto::ToolCall {
                    id: tc.id.clone(),
                    kind: tc.kind.clone(),
                    function: Some(proto::FunctionCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    }),
                })
                .collect()
        })
        .unwrap_or_default();
    proto::ChatMessage {
        role: m.role.clone(),
        content: m
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .map(str::to_string),
        tool_calls,
        tool_call_id: m.tool_call_id.clone(),
        name: m.name.clone(),
        refusal: m.refusal.clone(),
        provider_blocks_json: m
            .provider_blocks
            .as_ref()
            .and_then(|blocks| serde_json::to_string(blocks).ok()),
    }
}

#[must_use]
pub fn completion_outcome_to_proto(o: &CompletionOutcome) -> proto::CompletionOutcome {
    proto::CompletionOutcome {
        content: o.content.clone(),
        rounds: o.rounds,
        usage: o.usage.clone(),
        finish_reason: o.finish_reason.clone(),
    }
}

#[must_use]
pub fn response_outcome_to_proto(o: &responses::ResponseOutcome) -> proto::CompletionOutcome {
    proto::CompletionOutcome {
        content: o.content.clone(),
        rounds: o.rounds,
        usage: o.usage.clone(),
        finish_reason: None,
    }
}

fn chat_message_from_proto(p: proto::ChatMessage) -> ChatMessage {
    use crate::openai::{FunctionCall, MessageContent, ToolCall};
    let tool_calls = if p.tool_calls.is_empty() {
        None
    } else {
        Some(
            p.tool_calls
                .into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    kind: tc.kind,
                    function: FunctionCall {
                        name: tc
                            .function
                            .as_ref()
                            .map(|f| f.name.clone())
                            .unwrap_or_default(),
                        arguments: tc
                            .function
                            .as_ref()
                            .map(|f| f.arguments.clone())
                            .unwrap_or_else(|| "{}".to_string()),
                    },
                })
                .collect(),
        )
    };
    ChatMessage {
        role: p.role,
        content: p.content.map(MessageContent::Text),
        tool_calls,
        tool_call_id: p.tool_call_id,
        name: p.name,
        refusal: p.refusal,
        provider_blocks: p
            .provider_blocks_json
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok()),
    }
}

/// Rehydrate a [`ResumeContext`] from a stored [`proto::RunRecord`].
///
/// Caller must supply credentials separately (records never store secrets).
pub fn resume_context_from_record(
    record: &proto::RunRecord,
    mut options: ChatOptions,
) -> Result<ResumeContext, ResumeError> {
    let raw = record
        .options_json
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or(ResumeError::MissingOptions)?;
    let stored: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| ResumeError::InvalidOptions(e.to_string()))?;
    if let Some(model) = stored.get("model").and_then(|v| v.as_str()) {
        options.model = model.to_string();
    }
    if let Some(n) = stored.get("max_tool_rounds").and_then(|v| v.as_u64()) {
        options.max_tool_rounds = n as u32;
    }
    if let Some(sp) = stored.get("system_prompt").and_then(|v| v.as_str()) {
        options.system_prompt = Some(sp.to_string());
    }
    if let Some(re) = stored.get("reasoning_effort").and_then(|v| v.as_str()) {
        options.reasoning_effort = Some(re.to_string());
    }
    if let Some(extra) = stored.get("extra_json") {
        if extra.is_object() {
            options.extra_json = Some(extra.clone());
        }
    }
    if let Some(model) = &record.model_used {
        if !model.is_empty() {
            options.model = model.clone();
        }
    }

    Ok(ResumeContext {
        messages: record
            .messages
            .iter()
            .cloned()
            .map(chat_message_from_proto)
            .collect(),
        previous_response_id: record.previous_response_id.clone(),
        options,
    })
}

/// Serialize options (without secrets) for storage in audit records.
#[must_use]
pub fn options_snapshot(options: &ChatOptions) -> String {
    let mut snapshot = serde_json::json!({
        "model": options.model,
        "max_tool_rounds": options.max_tool_rounds,
        "system_prompt": options.system_prompt,
        "reasoning_effort": options.reasoning_effort,
        "reasoning_summary": options.reasoning_summary,
    });
    if let Some(extra) = &options.extra_json {
        snapshot["extra_json"] = extra.clone();
    }
    snapshot.to_string()
}

/// Export a run record as JSON for durable replay files.
pub fn export_record_json(record: &proto::RunRecord) -> String {
    format!(
        r#"{{"request_id":{},"options_json":{},"model_used":{},"previous_response_id":{},"started_at_ms":{},"finished_at_ms":{}}}"#,
        serde_json::to_string(&record.request_id).unwrap_or_default(),
        record
            .options_json
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default())
            .unwrap_or_else(|| "null".to_string()),
        record
            .model_used
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default())
            .unwrap_or_else(|| "null".to_string()),
        record
            .previous_response_id
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default())
            .unwrap_or_else(|| "null".to_string()),
        record.started_at_ms,
        record.finished_at_ms,
    )
}

/// Import a run record from JSON (metadata-only replay helper).
pub fn import_record_json(s: &str) -> Result<proto::RunRecord, serde_json::Error> {
    let v: serde_json::Value = serde_json::from_str(s)?;
    Ok(proto::RunRecord {
        request_id: v
            .get("request_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        messages: vec![],
        hook_events: vec![],
        outcome: None,
        started_at_ms: v.get("started_at_ms").and_then(|x| x.as_i64()).unwrap_or(0),
        finished_at_ms: v
            .get("finished_at_ms")
            .and_then(|x| x.as_i64())
            .unwrap_or(0),
        process_events: vec![],
        options_json: v
            .get("options_json")
            .and_then(|x| x.as_str())
            .map(str::to_string),
        model_used: v
            .get("model_used")
            .and_then(|x| x.as_str())
            .map(str::to_string),
        previous_response_id: v
            .get("previous_response_id")
            .and_then(|x| x.as_str())
            .map(str::to_string),
    })
}

/// Continue a chat run from a stored record.
pub async fn resume_chat(
    http: &HttpClient,
    registry: &crate::tools::ToolRegistry,
    hooks: &crate::hooks::HookRegistry,
    guardrails: &crate::guardrails::GuardrailRegistry,
    record: &proto::RunRecord,
    options: ChatOptions,
) -> Result<crate::chat::CompletionOutcome, crate::chat::ChatError> {
    let ctx = resume_context_from_record(record, options)
        .map_err(|e| crate::chat::ChatError::Api(format!("{e}")))?;
    complete_with_tools(
        http,
        registry,
        hooks,
        guardrails,
        ctx.messages,
        &ctx.options,
    )
    .await
}

/// Continue a Responses run from a stored record (uses last user message when present).
pub async fn resume_response(
    http: &HttpClient,
    registry: &crate::tools::ToolRegistry,
    hooks: &crate::hooks::HookRegistry,
    guardrails: &crate::guardrails::GuardrailRegistry,
    record: &proto::RunRecord,
    options: ChatOptions,
) -> Result<responses::ResponseOutcome, responses::ResponseError> {
    let ctx = resume_context_from_record(record, options)
        .map_err(|e| responses::ResponseError::Chat(crate::chat::ChatError::Api(format!("{e}"))))?;
    let mut opts = ctx.options;
    if let Some(id) = ctx.previous_response_id {
        let mut extra = match opts.extra_json.clone() {
            Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
            _ => json!({}),
        };
        if let serde_json::Value::Object(ref mut map) = extra {
            map.insert("previous_response_id".into(), json!(id));
        }
        opts.extra_json = Some(extra);
    }
    if !ctx.messages.is_empty()
        && ctx
            .messages
            .iter()
            .any(|m| m.role == "assistant" || m.role == "tool")
    {
        return responses::complete_from_messages(
            http,
            registry,
            hooks,
            guardrails,
            ctx.messages,
            &opts,
        )
        .await;
    }
    let user_message = ctx
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text())
        .unwrap_or("")
        .to_string();
    responses::complete_with_tools(http, registry, hooks, guardrails, user_message, &opts).await
}
