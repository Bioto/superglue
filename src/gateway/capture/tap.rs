//! Per-request capture helper used by gateway proxy paths.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde_json::Value;

use super::CaptureSink;
use super::record::{CAPTURE_SCHEMA, CaptureApi, CaptureRecord, CaptureUsage};
use crate::proto;

/// Inputs for starting a per-request capture tap.
pub struct CaptureTapContext {
    pub sink: Option<Arc<CaptureSink>>,
    pub request_id: String,
    pub user_id: String,
    pub key_id: Option<String>,
    pub api: CaptureApi,
    pub model_requested: String,
    pub stream: bool,
    pub max_response_bytes: usize,
}

pub struct CaptureTap {
    sink: Option<Arc<CaptureSink>>,
    request_id: String,
    user_id: String,
    key_id: Option<String>,
    api: CaptureApi,
    model_requested: String,
    model_resolved: Option<String>,
    stream: bool,
    max_response_bytes: usize,
    started_at: Instant,
    ts_start: String,
    request: Option<Value>,
    response: Option<Value>,
    sse: Vec<String>,
    sse_bytes: usize,
    sse_truncated: bool,
    finished: bool,
}

impl CaptureTap {
    #[must_use]
    pub fn begin(ctx: CaptureTapContext) -> Self {
        let CaptureTapContext {
            sink,
            request_id,
            user_id,
            key_id,
            api,
            model_requested,
            stream,
            max_response_bytes,
        } = ctx;
        let sink = sink.filter(|s| !s.should_exclude(&user_id));
        Self {
            sink,
            request_id,
            user_id,
            key_id,
            api,
            model_requested,
            model_resolved: None,
            stream,
            max_response_bytes,
            started_at: Instant::now(),
            ts_start: Utc::now().to_rfc3339(),
            request: None,
            response: None,
            sse: Vec::new(),
            sse_bytes: 0,
            sse_truncated: false,
            finished: false,
        }
    }

    pub fn set_request(&mut self, request: &Value) {
        if self.sink.is_some() {
            self.request = Some(request.clone());
        }
    }

    pub fn set_model_resolved(&mut self, model: String) {
        if self.sink.is_some() {
            self.model_resolved = Some(model);
        }
    }

    pub fn push_sse(&mut self, data: &str) {
        if self.sink.is_none() || self.sse_truncated {
            return;
        }
        let bytes = data.len();
        if self.sse_bytes.saturating_add(bytes) > self.max_response_bytes {
            self.sse_truncated = true;
            return;
        }
        self.sse_bytes = self.sse_bytes.saturating_add(bytes);
        self.sse.push(data.to_string());
    }

    pub fn set_response(&mut self, response: &Value) {
        if self.sink.is_some() {
            self.response = Some(response.clone());
        }
    }

    pub fn finish(&mut self, usage: Option<&proto::Usage>, cost_usd: Option<f64>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let Some(sink) = self.sink.take() else {
            return;
        };
        let request = self.request.take().unwrap_or(Value::Null);
        let record = CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: self.request_id.clone(),
            ts_start: self.ts_start.clone(),
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            user_id: self.user_id.clone(),
            key_id: self.key_id.clone(),
            api: self.api,
            model_requested: self.model_requested.clone(),
            model_resolved: self.model_resolved.clone(),
            stream: self.stream,
            request,
            response: self.response.take(),
            sse: std::mem::take(&mut self.sse),
            sse_truncated: self.sse_truncated,
            usage: usage.map(|u| CaptureUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            }),
            cost_usd,
            error: None,
        };
        sink.try_send(record);
    }

    pub fn finish_err(&mut self, message: impl Into<String>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let Some(sink) = self.sink.take() else {
            return;
        };
        let request = self.request.take().unwrap_or(Value::Null);
        let record = CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: self.request_id.clone(),
            ts_start: self.ts_start.clone(),
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            user_id: self.user_id.clone(),
            key_id: self.key_id.clone(),
            api: self.api,
            model_requested: self.model_requested.clone(),
            model_resolved: self.model_resolved.clone(),
            stream: self.stream,
            request,
            response: self.response.take(),
            sse: std::mem::take(&mut self.sse),
            sse_truncated: self.sse_truncated,
            usage: None,
            cost_usd: None,
            error: Some(message.into()),
        };
        sink.try_send(record);
    }
}
