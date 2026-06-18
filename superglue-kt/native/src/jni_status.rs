//! StatusEmitter JNI (`ProcessEventCallback`).

use std::sync::Arc;

use async_trait::async_trait;
use jni::objects::{GlobalRef, JClass, JObject, JValue};
use jni::sys::jlong;
use jni::JavaVM;
use jni::JNIEnv;
use superglue::events::{ProcessEvent, StatusEmitter, StatusSubscriber};

use crate::handle::{alloc, get_status_emitter, remove, KHandle};
use crate::jni_base::{ensure_jvm, jthrow, runtime};

pub fn process_event_json(event: &ProcessEvent) -> String {
    let usage = if let Some(u) = &event.usage {
        serde_json::json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    } else {
        serde_json::Value::Null
    };
    serde_json::json!({
        "kind": event.kind.as_str(),
        "request_id": event.request_id,
        "round": event.round,
        "model": event.model,
        "tool_call_count": event.tool_call_count,
        "usage": usage,
        "estimated_cost_usd": event.estimated_cost_usd,
        "error_type": event.error_type,
        "metadata": event.metadata,
        "timestamp_ms": event.timestamp_ms,
    })
    .to_string()
}

struct KotlinStatusSubscriber {
    jvm: Arc<JavaVM>,
    callback: GlobalRef,
}

#[async_trait]
impl StatusSubscriber for KotlinStatusSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        let json = process_event_json(&event);
        let jvm = Arc::clone(&self.jvm);
        let cb = self.callback.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().expect("attach");
            let j_arg = env.new_string(&json).expect("new_string");
            let _ = env.call_method(
                &cb,
                "onEvent",
                "(Ljava/lang/String;)V",
                &[JValue::Object(&j_arg)],
            );
        })
        .await;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_statusEmitterCreate(
    mut env: JNIEnv,
    _cl: JClass,
) -> jlong {
    let _ = ensure_jvm(&mut env);
    alloc(KHandle::StatusEmitter(Arc::new(StatusEmitter::new()))) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_statusEmitterDestroy(
    mut _env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) {
    remove(handle as u64);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_statusEmitterSubscribe(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    callback: JObject,
) {
    let jvm = ensure_jvm(&mut env);
    let emitter = match get_status_emitter(handle as u64) {
        Ok(e) => e,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let global = match env.new_global_ref(callback) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let subscriber = Arc::new(KotlinStatusSubscriber {
        jvm,
        callback: global,
    });
    runtime().block_on(
        emitter.subscribe(subscriber as Arc<dyn StatusSubscriber>),
    );
}
