use jni::JNIEnv;
use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
use std::sync::Arc;
use superglue::chat::complete_with_tools;
use superglue::openai::ChatMessage;

use crate::handle::ConversationState;
use crate::handle::KHandle;
use crate::handle::alloc;
use crate::handle::get_client;
use crate::handle::get_convo;
use crate::handle::remove;
use crate::jni_base::{
    completion_outcome_json, effective_http, jstr_from_str, jthrow, opt_i64, runtime,
};

use super::jni_client::{jstring_to_rust, opt_request_id};

fn messages_to_json(turns: &std::sync::Mutex<Vec<ChatMessage>>) -> String {
    let g = match turns.lock() {
        Ok(msgs) => match serde_json::to_value(&*msgs) {
            Ok(v) => v,
            Err(e) => return format!(r#"{{"error":"{e}"}}"#),
        },
        Err(e) => return format!(r#"{{"error":"{e}"}}"#),
    };
    g.to_string()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationFromClient(
    mut env: JNIEnv,
    _cl: JClass,
    client_handle: jlong,
) -> jlong {
    let cs = match get_client(client_handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return 0;
        }
    };
    let conv = ConversationState {
        state: Arc::clone(&cs),
        turns: Arc::new(std::sync::Mutex::new(vec![])),
    };
    alloc(KHandle::Conversation(conv)) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationDestroy(
    mut _env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) {
    remove(handle as u64);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationPushUser(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    text: JString,
) {
    let c = match get_convo(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let t = jstring_to_rust(&mut env, &text).unwrap_or_default();
    if let Ok(mut g) = c.turns.lock() {
        g.push(ChatMessage::text("user", t));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationPushAssistant(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    text: JString,
) {
    let c = match get_convo(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let t = jstring_to_rust(&mut env, &text).unwrap_or_default();
    if let Ok(mut g) = c.turns.lock() {
        g.push(ChatMessage::text("assistant", t));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationGetMessagesJson(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) -> jstring {
    let c = match get_convo(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &messages_to_json(c.turns.as_ref()))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_conversationComplete(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let c = match get_convo(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let msgs = match c.turns.lock() {
        Ok(g) => g.clone(),
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let http = match effective_http(
        &c.state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let reg = Arc::clone(&c.state.registry);
    let hooks = Arc::clone(&c.state.hooks);
    let gds = Arc::clone(&c.state.guardrails);
    let mut options = c.state.options.clone();
    options.request_id = opt_request_id(&mut env, &request_id);
    let outcome = match runtime().block_on(complete_with_tools(
        &http, &reg, &hooks, &gds, msgs, &options,
    )) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    if let Ok(mut g) = c.turns.lock() {
        *g = outcome.messages.clone();
    }
    jstr_from_str(&mut env, &completion_outcome_json(outcome))
}
