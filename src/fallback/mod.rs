//! Model fallback chains — try backup models when the primary fails after HTTP retries.

use std::sync::Arc;

use reqwest::StatusCode;
use serde_json::Value;

use crate::chat::ChatError;
use crate::events::{ProcessEvent, ProcessEventKind, StatusEmitter, emit_safe};
use crate::http::Error as HttpError;
use crate::http::RetryPolicy;

/// When to advance to the next model in a fallback chain (after per-request HTTP retries).
#[derive(Debug, Clone)]
pub struct FallbackPolicy {
    /// HTTP status codes that trigger fallback (default: transient overload / gateway errors).
    pub retryable_statuses: Vec<u16>,
    /// Retry on transport timeouts and connection failures.
    pub on_transport_error: bool,
    /// Retry on 401 / 403 when true (default false).
    pub on_auth_errors: bool,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self {
            retryable_statuses: vec![408, 425, 429, 500, 502, 503, 504],
            on_transport_error: true,
            on_auth_errors: false,
        }
    }
}

impl FallbackPolicy {
    pub fn allows_status(&self, status: StatusCode) -> bool {
        let code = status.as_u16();
        if self.on_auth_errors && matches!(code, 401 | 403) {
            return true;
        }
        self.retryable_statuses.contains(&code)
    }
}

/// Ordered list of models: primary first, then backups.
#[derive(Debug, Clone)]
pub struct ModelFallbackChain {
    pub models: Vec<String>,
    pub policy: FallbackPolicy,
}

impl ModelFallbackChain {
    pub fn new(models: Vec<String>) -> Self {
        Self {
            models,
            policy: FallbackPolicy::default(),
        }
    }

    pub fn with_policy(mut self, policy: FallbackPolicy) -> Self {
        self.policy = policy;
        self
    }
}

/// Models to attempt: explicit chain if set, otherwise `[primary_model]`.
#[must_use]
pub fn effective_models(primary_model: &str, chain: Option<&ModelFallbackChain>) -> Vec<String> {
    match chain {
        Some(c) if !c.models.is_empty() => c.models.clone(),
        _ => vec![primary_model.to_string()],
    }
}

/// Whether an HTTP error should trigger trying the next fallback model.
#[must_use]
pub fn http_error_eligible_for_fallback(err: &HttpError, policy: &FallbackPolicy) -> bool {
    match err {
        HttpError::Reqwest(e) if policy.on_transport_error => {
            e.is_timeout() || e.is_connect() || RetryPolicy::is_retryable_post_reqwest_error(e)
        }
        HttpError::Unsuccessful { status, .. } => policy.allows_status(*status),
        HttpError::RetriesExhausted { last, .. } => last
            .map(|s| policy.allows_status(s))
            .unwrap_or(policy.on_transport_error),
        _ => false,
    }
}

#[must_use]
pub fn chat_error_eligible_for_fallback(err: &ChatError, policy: &FallbackPolicy) -> bool {
    match err {
        ChatError::Http(e) => http_error_eligible_for_fallback(e, policy),
        _ => false,
    }
}

/// Patch `model` on a JSON request body.
pub fn set_body_model(body: &mut Value, model: &str) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
}

/// Emit status metadata when switching models.
pub async fn emit_model_fallback(
    emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    from: &str,
    to: &str,
    round: u32,
) {
    if let Some(emitter) = emitter {
        let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallError, request_id, to);
        ev.round = round;
        ev.error_type = Some("model_fallback".to_string());
        ev.metadata
            .insert("fallback_from".to_string(), from.to_string());
        ev.metadata
            .insert("fallback_to".to_string(), to.to_string());
        emit_safe(Some(emitter), ev).await;
    }
}
