// Client-related JNI. Names match `com.superglue.kt.SuperglueNativeJni` (Java).

use std::sync::Arc;

use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use serde_json::json;
use serde_json::Value;
use superglue::agents::AgentEngine as SgAgentEngine;
use superglue::batch::{BatchConfig, BatchRequest, ErrorStrategy, batch_complete};
use superglue::chat::{
    ChatOptions, complete_with_tools, stream_complete, stream_complete_with_tools,
};
use superglue::mcp::{McpHttpConfig, McpSession, McpStdioConfig};
use superglue::responses::{
    complete_with_tools as complete_response_with_tools,
    stream_response as stream_response_api,
};
use superglue::guardrails::{
    BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailHandler, GuardrailStage,
    LengthStrategy, MaxLengthGuardrail, PiiRedactGuardrail,
};
use superglue::hooks::{HookConfig, HookErrorStrategy, HookStage};
use superglue::openai::ChatMessage;
use superglue::tools::{Tool, ToolSpec};

use crate::adapters::{GuardrailStateFilter, KotlinGuardrail, KotlinHook, KotlinJsonTool};
use crate::config::apply_status_emitter;
use crate::config::build_client_state;
use crate::config::parse_agent_spec;
use crate::config::value_from_str;
use crate::handle::alloc;
use crate::handle::get_client;
use crate::handle::get_status_emitter;
use crate::handle::remove;
use crate::handle::KHandle;
use crate::jni_base::{
    completion_outcome_json, ensure_jvm, effective_http, jstr_from_str, jthrow, opt_i64, runtime,
    response_outcome_json, response_stream_outcome_json, stream_outcome_json,
};

pub(crate) fn jstring_to_rust(env: &mut JNIEnv, s: &JString) -> Result<String, jni::errors::Error> {
    let js = env.get_string(s)?;
    Ok(String::from(js))
}

pub(crate) fn opt_request_id(_env: &mut JNIEnv, request_id: &JString) -> Option<String> {
    // Empty string means None (callers use "" for absent).
    jstring_to_rust(_env, request_id).ok().filter(|s| !s.is_empty())
}

pub(crate) fn opt_reasoning_effort(env: &mut JNIEnv, effort: &JString) -> Option<String> {
    jstring_to_rust(env, effort).ok().filter(|s| !s.is_empty())
}

fn finalize_call_options(
    base: &ChatOptions,
    request_id: Option<String>,
    reasoning_effort: Option<String>,
) -> ChatOptions {
    let mut options = base.clone();
    options.request_id = request_id;
    if let Some(re) = reasoning_effort {
        options.reasoning_effort = Some(re);
    }
    options
}

// --- JNI ---

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientCreate(
    mut env: JNIEnv,
    _cl: JClass,
    config: JString,
) -> jlong {
    let _ = ensure_jvm(&mut env);
    let c: String = match jstring_to_rust(&mut env, &config) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return 0;
        }
    };
    let state = match build_client_state(&c) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return 0;
        }
    };
    alloc(KHandle::Client(Arc::new(state))) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientCreateWithEmitter(
    mut env: JNIEnv,
    _cl: JClass,
    config: JString,
    emitter_handle: jlong,
) -> jlong {
    let _ = ensure_jvm(&mut env);
    let c: String = match jstring_to_rust(&mut env, &config) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return 0;
        }
    };
    let mut state = match build_client_state(&c) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return 0;
        }
    };
    if emitter_handle != 0 {
        let emitter = match get_status_emitter(emitter_handle as u64) {
            Ok(e) => e,
            Err(e) => {
                jthrow(&mut env, &e);
                return 0;
            }
        };
        apply_status_emitter(&mut state, emitter);
    }
    alloc(KHandle::Client(Arc::new(state))) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientDestroy(
    mut _env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) {
    remove(handle as u64);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientCancel(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let next = state.cancel_tx.borrow().saturating_add(1);
    let _ = state.cancel_tx.send(next);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientRegisterTool(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    name: JString,
    desc: JString,
    parameters_json: JString,
    callback: JObject,
) {
    let jvm = ensure_jvm(&mut env);
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let name = match jstring_to_rust(&mut env, &name) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let desc = match jstring_to_rust(&mut env, &desc) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let pjson = match jstring_to_rust(&mut env, &parameters_json) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let parameters: Value = match serde_json::from_str(&pjson) {
        Ok(v) => v,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let g = match env.new_global_ref(&callback) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let spec = ToolSpec {
        name,
        description: Some(desc),
        parameters_schema: parameters,
    };
    let tool: Arc<dyn Tool> = Arc::new(KotlinJsonTool {
        spec,
        jvm: jvm.clone(),
        callback: g,
    });
    if let Err(e) = runtime().block_on(state.registry.register(tool)) {
        jthrow(&mut env, &e.to_string());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientRegisterHook(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    stage: JString,
    name: JString,
    error_strategy: JString,
    callback: JObject,
) {
    let jvm = ensure_jvm(&mut env);
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let stage_s = jstring_to_rust(&mut env, &stage).unwrap_or_default();
    let hook_stage = match HookStage::from_str(&stage_s) {
        Some(s) => s,
        None => {
            jthrow(
                &mut env,
                "invalid hook stage: expected pre_completion, post_completion, pre_tool, post_tool, on_retry, pre_batch_item, post_batch_item",
            );
            return;
        }
    };
    let name_s = jstring_to_rust(&mut env, &name).unwrap_or_default();
    let name_final = if name_s.is_empty() {
        "hook".to_string()
    } else {
        name_s
    };
    let es = jstring_to_rust(&mut env, &error_strategy).unwrap_or_default();
    let strategy = match HookErrorStrategy::from_str(if es.is_empty() {
        "skip"
    } else {
        &es
    }) {
        Some(s) => s,
        None => {
            jthrow(&mut env, "invalid error_strategy; expected 'skip' or 'abort'");
            return;
        }
    };
    let g = match env.new_global_ref(&callback) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let hook = KotlinHook {
        jvm: jvm.clone(),
        callback: g,
        name: name_final.clone(),
    };
    let config = HookConfig {
        name: name_final,
        error_strategy: strategy,
        handler: Arc::new(hook),
    };
    runtime().block_on(state.hooks.add(hook_stage, config));
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientRegisterGuardrail(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    callback: JObject,
    stage: JString,
    name: JString,
) {
    let jvm = ensure_jvm(&mut env);
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let stage_s = jstring_to_rust(&mut env, &stage).unwrap_or_else(|_| "both".to_string());
    let stages: Vec<GuardrailStage> = match stage_s.as_str() {
        "input" => vec![GuardrailStage::Input],
        "output" => vec![GuardrailStage::Output],
        "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
        _ => {
            jthrow(
                &mut env,
                "invalid guardrail stage: expected 'input', 'output', or 'both'",
            );
            return;
        }
    };
    let name_s = jstring_to_rust(&mut env, &name).unwrap_or_else(|_| "guardrail".to_string());
    let name_f = if name_s.is_empty() {
        "guardrail".to_string()
    } else {
        name_s
    };
    let g = match env.new_global_ref(&callback) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    let gds = Arc::clone(&state.guardrails);
    let name_c = name_f.clone();
    let jv = jvm;
    runtime().block_on(async move {
        for s in &stages {
            let filter = match s {
                GuardrailStage::Input => GuardrailStateFilter::Input,
                GuardrailStage::Output => GuardrailStateFilter::Output,
            };
            let kg = Arc::new(KotlinGuardrail {
                jvm: Arc::clone(&jv),
                callback: g.clone(),
                stage_filter: filter,
                name: name_c.clone(),
            }) as Arc<dyn GuardrailHandler>;
            let cfg = GuardrailConfig {
                name: name_c.clone(),
                handler: kg,
            };
            match s {
                GuardrailStage::Input => gds.add_input(cfg).await,
                GuardrailStage::Output => gds.add_output(cfg).await,
            }
        }
    });
}

// --- add_blocklist, add_max_length, add_pii, runAgent, complete, completeMessages, stream, batch

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientAddBlocklistGuardrail(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    patterns_json: JString,
    action: JString,
    stage: JString,
    name: JString,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let patterns: Vec<String> = match serde_json::from_str(
        &jstring_to_rust(&mut env, &patterns_json).unwrap_or_default(),
    ) {
        Ok(p) => p,
        Err(e) => {
            jthrow(&mut env, &format!("patterns json: {e}"));
            return;
        }
    };
    let bl_action = match jstring_to_rust(&mut env, &action)
        .unwrap_or_default()
        .as_str()
    {
        "" | "block" => BlocklistAction::Block,
        "redact" => BlocklistAction::Redact,
        o => {
            jthrow(
                &mut env,
                &format!("invalid blocklist action {o:?}; expected 'block' or 'redact'"),
            );
            return;
        }
    };
    let stage_s = jstring_to_rust(&mut env, &stage).unwrap_or_else(|_| "both".to_string());
    let stages: Vec<GuardrailStage> = match stage_s.as_str() {
        "input" => vec![GuardrailStage::Input],
        "output" => vec![GuardrailStage::Output],
        "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
        _ => {
            jthrow(
                &mut env,
                "invalid stage: expected 'input', 'output', or 'both'",
            );
            return;
        }
    };
    let name_str = jstring_to_rust(&mut env, &name).unwrap_or_else(|_| "blocklist".to_string());
    let patterns_ref: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
    let gds = Arc::clone(&state.guardrails);
    if let Err(e) = runtime().block_on(async move {
        if stages.contains(&GuardrailStage::Input) {
            let h_in = match BlocklistGuardrail::new(&patterns_ref, bl_action) {
                Ok(b) => b.for_stages(vec![GuardrailStage::Input]),
                Err(e) => return Err(e.to_string()),
            };
            gds
                .add_input(GuardrailConfig {
                    name: name_str.clone(),
                    handler: Arc::new(h_in),
                })
                .await;
        }
        if stages.contains(&GuardrailStage::Output) {
            let h_out = match BlocklistGuardrail::new(&patterns_ref, bl_action) {
                Ok(b) => b.for_stages(vec![GuardrailStage::Output]),
                Err(e) => return Err(e.to_string()),
            };
            gds
                .add_output(GuardrailConfig {
                    name: name_str,
                    handler: Arc::new(h_out),
                })
                .await;
        }
        Ok::<(), String>(())
    }) {
        jthrow(&mut env, &e);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientAddMaxLengthGuardrail(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    max_input: i32,
    max_output: i32,
    strategy: JString,
    name: JString,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let len_strategy = match jstring_to_rust(&mut env, &strategy)
        .unwrap_or_default()
        .as_str()
    {
        "" | "block" => LengthStrategy::Block,
        "truncate" => LengthStrategy::Truncate,
        o => {
            jthrow(
                &mut env,
                &format!("invalid strategy {o:?}; expected 'block' or 'truncate'"),
            );
            return;
        }
    };
    let max_in = if max_input < 0 {
        None
    } else {
        Some(max_input as usize)
    };
    let max_out = if max_output < 0 {
        None
    } else {
        Some(max_output as usize)
    };
    let handler: Arc<dyn superglue::guardrails::GuardrailHandler> = Arc::new(
        MaxLengthGuardrail::new(max_in, max_out, len_strategy),
    );
    let name_s = jstring_to_rust(&mut env, &name).unwrap_or_else(|_| "max_length".to_string());
    let gds = Arc::clone(&state.guardrails);
    runtime().block_on(async move {
        if max_in.is_some() {
            gds
                .add_input(GuardrailConfig {
                    name: name_s.clone(),
                    handler: Arc::clone(&handler),
                })
                .await;
        }
        if max_out.is_some() {
            gds
                .add_output(GuardrailConfig {
                    name: name_s,
                    handler,
                })
                .await;
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientAddPiiGuardrail(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    stage: JString,
    name: JString,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let stage_s = jstring_to_rust(&mut env, &stage).unwrap_or_else(|_| "both".to_string());
    let stages: Vec<GuardrailStage> = match stage_s.as_str() {
        "input" => vec![GuardrailStage::Input],
        "output" => vec![GuardrailStage::Output],
        "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
        _ => {
            jthrow(
                &mut env,
                "invalid stage: expected 'input', 'output', or 'both'",
            );
            return;
        }
    };
    let name_s = jstring_to_rust(&mut env, &name).unwrap_or_else(|_| "pii_redact".to_string());
    let gds = Arc::clone(&state.guardrails);
    runtime().block_on(async move {
        if stages.contains(&GuardrailStage::Input) {
            gds
                .add_input(GuardrailConfig {
                    name: name_s.clone(),
                    handler: Arc::new(
                        PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Input]),
                    ),
                })
                .await;
        }
        if stages.contains(&GuardrailStage::Output) {
            gds
                .add_output(GuardrailConfig {
                    name: name_s,
                    handler: Arc::new(
                        PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Output]),
                    ),
                })
                .await;
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientRunAgent(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    spec_json: JString,
    user_message: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let spec = match parse_agent_spec(
        &jstring_to_rust(&mut env, &spec_json).unwrap_or_default(),
    ) {
        Ok(s) => s,
        Err(err) => {
            jthrow(&mut env, &err);
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let reg = Arc::clone(&state.registry);
    let hooks = Arc::clone(&state.hooks);
    let gds = Arc::clone(&state.guardrails);
    let mut opts = state.options.clone();
    let rid = opt_request_id(&mut env, &request_id);
    opts.request_id = rid;
    if !spec.model.is_empty() {
        opts.model = spec.model.clone();
    }
    opts.max_tool_rounds = spec.max_tool_rounds;
    let engine = SgAgentEngine::new(spec)
        .with_hooks(hooks)
        .with_guardrails(gds);
    let outcome = match runtime().block_on(
        engine.run(&http, &reg, user_message, &opts),
    ) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &completion_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientComplete(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    user_message: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
    reasoning_effort: JString,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let messages = vec![ChatMessage::text("user", user_message)];
    let reg = Arc::clone(&state.registry);
    let hooks = Arc::clone(&state.hooks);
    let gds = Arc::clone(&state.guardrails);
    let options = finalize_call_options(
        &state.options,
        opt_request_id(&mut env, &request_id),
        opt_reasoning_effort(&mut env, &reasoning_effort),
    );
    let mut cancel_rx = state.cancel_tx.subscribe();
    let cancel_generation = *cancel_rx.borrow();
    let outcome = match runtime().block_on(async {
        tokio::select! {
            outcome = complete_with_tools(&http, &reg, &hooks, &gds, messages, &options) => {
                outcome.map_err(|e| e.to_string())
            }
            _ = wait_for_cancel(&mut cancel_rx, cancel_generation) => {
                Err("superglue execution cancelled".to_string())
            }
        }
    }) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &completion_outcome_json(outcome))
}

async fn wait_for_cancel(rx: &mut tokio::sync::watch::Receiver<u64>, generation: u64) {
    loop {
        if *rx.borrow() != generation {
            return;
        }
        if rx.changed().await.is_err() {
            return;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientCompleteMessages(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    messages_json: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let mj = jstring_to_rust(&mut env, &messages_json).unwrap_or_default();
    let v: Value = match value_from_str(&mj) {
        Ok(v) => v,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let rust_messages: Vec<ChatMessage> = match serde_json::from_value(v) {
        Ok(m) => m,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let reg = Arc::clone(&state.registry);
    let hooks = Arc::clone(&state.hooks);
    let gds = Arc::clone(&state.guardrails);
    let mut options = state.options.clone();
    options.request_id = opt_request_id(&mut env, &request_id);
    let outcome = match runtime().block_on(complete_with_tools(
        &http, &reg, &hooks, &gds, rust_messages, &options,
    )) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &completion_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientStream(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    user_message: JString,
    on_token: JObject,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let jvm = ensure_jvm(&mut env);
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let g_cb = match env.new_global_ref(&on_token) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let mut options = state.options.clone();
    options.request_id = opt_request_id(&mut env, &request_id);
    let messages = vec![ChatMessage::text("user", user_message)];
    let hooks = Arc::clone(&state.hooks);
    let guardrails = Arc::clone(&state.guardrails);
    let registry = Arc::clone(&state.registry);
    let has_tools = !runtime().block_on(registry.list_specs()).is_empty();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let jvm2 = jvm.clone();
    let g2 = g_cb.clone();
    let consumer = std::thread::spawn(move || {
        let jvm2 = jvm2;
        let g2 = g2;
        if let Ok(mut env) = jvm2.attach_current_thread() {
            while let Some(delta) = rx.blocking_recv() {
                if let Ok(jdelta) = env.new_string(&delta) {
                    let _ = env.call_method(
                        &g2,
                        "onToken",
                        "(Ljava/lang/String;)V",
                        &[JValue::Object(&jdelta)],
                    );
                }
            }
        }
    });
    let on_delta = move |delta: String| {
        let _ = tx.send(delta);
    };
    let stream_result = if has_tools {
        runtime()
            .block_on(stream_complete_with_tools(
                &http,
                &registry,
                &hooks,
                &guardrails,
                messages,
                &options,
                on_delta,
            ))
            .map(|o| superglue::chat::StreamOutcome {
                content: o.content,
                finish_reason: o.finish_reason,
                usage: o.usage,
                request_id: o.request_id,
            })
    } else {
        runtime().block_on(stream_complete(
            &http,
            &hooks,
            &guardrails,
            messages,
            &options,
            on_delta,
        ))
    };
    let _ = consumer.join();
    let outcome = match stream_result {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &stream_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientBatch(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    requests_json: JString,
    max_concurrent: i32,
    error_strategy: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let rj = jstring_to_rust(&mut env, &requests_json).unwrap_or_default();
    let batch_requests: Vec<BatchRequest> = match serde_json::from_str::<
        Vec<serde_json::Value>,
    >(&rj)
    {
        Ok(vs) => {
            let mut out = vec![];
            for v in vs {
                let id = v.get("id").and_then(|x| x.as_str()).map(String::from);
                let sp = v
                    .get("systemPrompt")
                    .and_then(|x| x.as_str())
                    .map(String::from);
                let prompt = v
                    .get("prompt")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut br = BatchRequest::new(prompt);
                br.id = id;
                br.system_prompt = sp;
                out.push(br);
            }
            out
        }
        Err(e) => {
            jthrow(&mut env, &format!("batch requests: {e}"));
            return std::ptr::null_mut();
        }
    };
    let strategy = match ErrorStrategy::from_str(
        &jstring_to_rust(&mut env, &error_strategy).unwrap_or_else(|_| "continue".to_string()),
    ) {
        Some(s) => s,
        None => {
            jthrow(
                &mut env,
                "invalid error_strategy: expected 'continue', 'skip', or 'fail_fast'",
            );
            return std::ptr::null_mut();
        }
    };
    let config = BatchConfig {
        max_concurrent: (if max_concurrent < 0 { 5 } else { max_concurrent }) as usize,
        error_strategy: strategy,
        timeout: opt_i64(timeout_secs)
            .map(|s| std::time::Duration::from_secs(s.max(0) as u64)),
        connect_timeout: opt_i64(connect_timeout_secs)
            .map(|s| std::time::Duration::from_secs(s.max(0) as u64)),
        cancel: None,
    };
    let http = Arc::clone(&state.http);
    let reg = Arc::clone(&state.registry);
    let hooks = Arc::clone(&state.hooks);
    let gds = Arc::clone(&state.guardrails);
    let options = state.options.clone();
    let response = match runtime().block_on(batch_complete(
        http, reg, hooks, gds, batch_requests, &options, config,
    )) {
        Ok(r) => r,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let results: Vec<_> = response
        .results
        .into_iter()
        .map(|r| {
            let usage = r.usage.map(|u| {
                json!({
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                })
            });
            json!({
                "id": r.id,
                "success": r.success,
                "content": r.content,
                "error": r.error,
                "rounds": r.rounds,
                "usage": usage,
                "elapsedSecs": r.elapsed_secs,
            })
        })
        .collect();
    let total_usage = response.total_usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });
    let v = json!({
        "results": results,
        "totalRequests": response.total_requests,
        "successful": response.successful,
        "failed": response.failed,
        "elapsedSecs": response.elapsed_secs,
        "totalUsage": total_usage,
    });
    jstr_from_str(&mut env, &v.to_string())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientCompleteResponse(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    user_message: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
    reasoning_effort: JString,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let reg = Arc::clone(&state.registry);
    let hooks = Arc::clone(&state.hooks);
    let gds = Arc::clone(&state.guardrails);
    let options = finalize_call_options(
        &state.options,
        opt_request_id(&mut env, &request_id),
        opt_reasoning_effort(&mut env, &reasoning_effort),
    );
    let outcome = match runtime().block_on(complete_response_with_tools(
        &http,
        &reg,
        &hooks,
        &gds,
        user_message,
        &options,
    )) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &response_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientStreamResponse(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    user_message: JString,
    on_token: JObject,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let jvm = ensure_jvm(&mut env);
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let g_cb = match env.new_global_ref(&on_token) {
        Ok(g) => g,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &state.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let mut options = state.options.clone();
    options.request_id = opt_request_id(&mut env, &request_id);
    let hooks = Arc::clone(&state.hooks);
    let guardrails = Arc::clone(&state.guardrails);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let jvm2 = jvm.clone();
    let g2 = g_cb.clone();
    let consumer = std::thread::spawn(move || {
        if let Ok(mut env) = jvm2.attach_current_thread() {
            while let Some(delta) = rx.blocking_recv() {
                if let Ok(jdelta) = env.new_string(&delta) {
                    let _ = env.call_method(
                        &g2,
                        "onToken",
                        "(Ljava/lang/String;)V",
                        &[JValue::Object(&jdelta)],
                    );
                }
            }
        }
    });
    let on_delta = move |delta: String| {
        let _ = tx.send(delta);
    };
    let stream_result = runtime().block_on(stream_response_api(
        &http,
        &hooks,
        &guardrails,
        user_message,
        &options,
        on_delta,
    ));
    let _ = consumer.join();
    let outcome = match stream_result {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &response_stream_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientConnectMcpStdio(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    command: JString,
    args_json: JString,
    env_json: JString,
    prefix: JString,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let command = jstring_to_rust(&mut env, &command).unwrap_or_default();
    let args_raw = jstring_to_rust(&mut env, &args_json).unwrap_or_else(|_| "[]".to_string());
    let args: Vec<String> = match value_from_str(&args_raw) {
        Ok(v) => serde_json::from_value(v).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let env_map: Option<std::collections::HashMap<String, String>> = {
        let raw = jstring_to_rust(&mut env, &env_json).unwrap_or_default();
        if raw.is_empty() {
            None
        } else {
            value_from_str(&raw)
                .ok()
                .and_then(|v| serde_json::from_value(v).ok())
        }
    };
    let prefix = opt_request_id(&mut env, &prefix);
    let reg = Arc::clone(&state.registry);
    let session = match runtime().block_on(McpSession::connect_stdio(McpStdioConfig {
        command,
        args,
        env: env_map,
        label: prefix.clone(),
    })) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    if let Err(e) = runtime().block_on(session.register_tools(&reg, prefix.as_deref())) {
        jthrow(&mut env, &e.to_string());
        return;
    }
    state.mcp_sessions.lock().unwrap().push(session);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientConnectMcpHttp(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    url: JString,
    prefix: JString,
) {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let url = jstring_to_rust(&mut env, &url).unwrap_or_default();
    let prefix = opt_request_id(&mut env, &prefix);
    let reg = Arc::clone(&state.registry);
    let session = match runtime().block_on(McpSession::connect_http(McpHttpConfig {
        url,
        label: prefix.clone(),
        auth_header: None,
        custom_headers: Default::default(),
    })) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return;
        }
    };
    if let Err(e) = runtime().block_on(session.register_tools(&reg, prefix.as_deref())) {
        jthrow(&mut env, &e.to_string());
        return;
    }
    state.mcp_sessions.lock().unwrap().push(session);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientUploadFile(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    path: JString,
    purpose: JString,
    provider: JString,
) -> jstring {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let path = jstring_to_rust(&mut env, &path).unwrap_or_default();
    let purpose_str = jstring_to_rust(&mut env, &purpose).unwrap_or_else(|_| "user_data".to_string());
    let purpose = match purpose_str.as_str() {
        "assistants" => superglue::files::FilePurpose::Assistants,
        "user_data" => superglue::files::FilePurpose::UserData,
        "batch" => superglue::files::FilePurpose::Batch,
        other => {
            jthrow(&mut env, &format!("unknown file purpose: {other}"));
            return std::ptr::null_mut();
        }
    };
    let provider_opt = opt_request_id(&mut env, &provider);
    let provider_id = provider_opt
        .as_deref()
        .and_then(superglue::client::provider_id_from_str)
        .unwrap_or_else(|| superglue::providers::parse_model_ref(&state.options.model).provider);
    let creds = superglue::chat::credentials_for(&state.options);
    let uploaded = match runtime().block_on(superglue::files::upload_file(
        &state.http,
        &creds,
        provider_id,
        std::path::Path::new(&path),
        purpose,
        state.max_upload_bytes,
    )) {
        Ok(u) => u,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let json = json!({
        "provider": uploaded.provider.to_string(),
        "fileId": uploaded.file_id,
        "filename": uploaded.filename,
        "bytes": uploaded.bytes,
        "purpose": uploaded.purpose.as_openai_str(),
    });
    jstr_from_str(&mut env, &json.to_string())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientMessageWithFileBytes(
    mut env: JNIEnv,
    _cl: JClass,
    filename: JString,
    file_bytes: jni::objects::JByteArray,
    text: JString,
) -> jstring {
    let filename = jstring_to_rust(&mut env, &filename).unwrap_or_default();
    let text_opt = opt_request_id(&mut env, &text);
    let bytes: Vec<u8> = match env.convert_byte_array(&file_bytes) {
        Ok(b) => b,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    let msg =
        superglue::files::message_with_file_bytes(text_opt.as_deref(), &filename, &bytes);
    let json = match serde_json::to_string(&msg) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &json)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientToolsRegistryPtr(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) -> jlong {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return 0;
        }
    };
    Arc::as_ptr(&state.registry) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_clientHooksRegistryPtr(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) -> jlong {
    let state = match get_client(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return 0;
        }
    };
    Arc::as_ptr(&state.hooks) as jlong
}