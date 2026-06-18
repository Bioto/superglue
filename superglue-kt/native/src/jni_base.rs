//! Shared helpers for JNI modules.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use jni::sys::jstring;
use jni::JavaVM;
use jni::JNIEnv;
use serde_json::json;
use superglue::http::HttpClient;
use tokio::runtime::Runtime;

pub static RT: OnceLock<Runtime> = OnceLock::new();
pub static JVM: OnceLock<Arc<JavaVM>> = OnceLock::new();

pub fn runtime() -> &'static Runtime {
    RT.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

pub fn ensure_jvm(env: &mut JNIEnv) -> Arc<JavaVM> {
    JVM.get_or_init(|| {
        let vm = env.get_java_vm().expect("get_java_vm");
        Arc::new(vm)
    })
    .clone()
}

pub fn jthrow(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

pub fn jstr_from_str(env: &mut JNIEnv, s: &str) -> jstring {
    match env.new_string(s) {
        Ok(js) => js.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Map per-request `long` timeout: `-1` = use client default.
pub fn opt_i64(n: i64) -> Option<i64> {
    if n < 0 {
        None
    } else {
        Some(n)
    }
}

pub fn effective_http(
    base: &Arc<HttpClient>,
    timeout_secs: Option<i64>,
    connect_timeout_secs: Option<i64>,
) -> Result<Arc<HttpClient>, String> {
    if timeout_secs.is_none() && connect_timeout_secs.is_none() {
        return Ok(Arc::clone(base));
    }
    let t = timeout_secs
        .map(|s| Duration::from_secs(s.max(0) as u64))
        .unwrap_or(base.config.timeout);
    let ct = connect_timeout_secs
        .map(|s| Duration::from_secs(s.max(0) as u64))
        .unwrap_or(base.config.connect_timeout);
    base
        .clone_with_timeouts(t, ct)
        .map(Arc::new)
        .map_err(|e| e.to_string())
}

pub fn completion_outcome_json(outcome: superglue::chat::CompletionOutcome) -> String {
    let u = outcome.usage.map(|o| {
        json!({
            "prompt_tokens": o.prompt_tokens,
            "completion_tokens": o.completion_tokens,
            "total_tokens": o.total_tokens,
        })
    });
    let messages = match serde_json::to_value(&outcome.messages) {
        Ok(m) => m,
        Err(e) => return format!(r#"{{"error":"messages: {e}"}}"#),
    };
    let v = json!({
        "content": outcome.content,
        "rounds": outcome.rounds,
        "usage": u,
        "requestId": outcome.request_id,
        "modelUsed": outcome.model_used,
        "messages": messages,
    });
    v.to_string()
}

pub fn response_outcome_json(outcome: superglue::responses::ResponseOutcome) -> String {
    let u = outcome.usage.map(|o| {
        json!({
            "prompt_tokens": o.prompt_tokens,
            "completion_tokens": o.completion_tokens,
            "total_tokens": o.total_tokens,
        })
    });
    let v = json!({
        "id": outcome.id,
        "content": outcome.content,
        "rounds": outcome.rounds,
        "usage": u,
        "requestId": outcome.request_id,
        "modelUsed": outcome.model_used,
    });
    v.to_string()
}

pub fn response_stream_outcome_json(o: superglue::responses::ResponseStreamOutcome) -> String {
    let u = o.usage.map(|x| {
        json!({
            "prompt_tokens": x.prompt_tokens,
            "completion_tokens": x.completion_tokens,
            "total_tokens": x.total_tokens,
        })
    });
    let v = json!({
        "id": o.id,
        "content": o.content,
        "usage": u,
        "requestId": o.request_id,
        "modelUsed": o.model_used,
    });
    v.to_string()
}

pub fn stream_outcome_json(o: superglue::chat::StreamOutcome) -> String {
    let u = o.usage.map(|x| {
        json!({
            "prompt_tokens": x.prompt_tokens,
            "completion_tokens": x.completion_tokens,
            "total_tokens": x.total_tokens,
        })
    });
    let v = json!({
        "content": o.content,
        "finishReason": o.finish_reason,
        "usage": u,
        "requestId": o.request_id,
    });
    v.to_string()
}
