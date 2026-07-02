use std::sync::Arc;

use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jboolean, jlong, jstring};
use jni::JNIEnv;
use serde_json::Value;
use superglue::agents::AgentEngine as SgAgentEngine;
use superglue::guardrails::GuardrailConfig;
use superglue::guardrails::GuardrailHandler;
use superglue::guardrails::GuardrailStage;
use superglue::hooks::HookConfig;
use superglue::hooks::HookErrorStrategy;
use superglue::hooks::HookStage;
use superglue::tools::Tool;
use superglue::tools::ToolSpec;

use crate::adapters::{GuardrailStateFilter, KotlinGuardrail, KotlinHook, KotlinJsonTool};
use crate::config::build_engine_state;
use crate::handle::alloc;
use crate::handle::get_engine;
use crate::handle::remove;
use crate::handle::KHandle;
use crate::jni_base::{
    completion_outcome_json, ensure_jvm, effective_http, jstr_from_str, jthrow, opt_i64, runtime,
    stream_outcome_json,
};

use super::jni_client::jstring_to_rust;
use super::jni_client::opt_request_id;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineCreate(
    mut env: JNIEnv,
    _cl: JClass,
    spec_json: JString,
    http_config_json: JString,
) -> jlong {
    let _ = ensure_jvm(&mut env);
    let spec = jstring_to_rust(&mut env, &spec_json).unwrap_or_default();
    let httpc = jstring_to_rust(&mut env, &http_config_json).unwrap_or_default();
    let e = match build_engine_state(&spec, &httpc) {
        Ok(s) => s,
        Err(err) => {
            jthrow(&mut env, &err);
            return 0;
        }
    };
    alloc(KHandle::Engine(Arc::new(e))) as jlong
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineDestroy(
    mut _env: JNIEnv,
    _cl: JClass,
    handle: jlong,
) {
    remove(handle as u64);
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineRegisterTool(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    name: JString,
    desc: JString,
    parameters_json: JString,
    static_tool: jboolean,
    callback: JObject,
) {
    let jvm = ensure_jvm(&mut env);
    let eng = match get_engine(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let name = jstring_to_rust(&mut env, &name).unwrap_or_default();
    let d = jstring_to_rust(&mut env, &desc).unwrap_or_default();
    let pjson = jstring_to_rust(&mut env, &parameters_json).unwrap_or_default();
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
        description: Some(d),
        parameters_schema: parameters,
        static_tool: static_tool != 0,
    };
    let tool: Arc<dyn Tool> = Arc::new(KotlinJsonTool {
        spec,
        jvm: jvm.clone(),
        callback: g,
    });
    if let Err(e) = runtime().block_on(eng.registry.register(tool)) {
        jthrow(&mut env, &e.to_string());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineRegisterHook(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    stage: JString,
    name: JString,
    error_strategy: JString,
    callback: JObject,
) {
    let jvm = ensure_jvm(&mut env);
    let eng = match get_engine(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return;
        }
    };
    let stage_s = jstring_to_rust(&mut env, &stage).unwrap_or_default();
    let hook_st = match HookStage::from_str(&stage_s) {
        Some(s) => s,
        None => {
            jthrow(&mut env, "invalid hook stage");
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
            jthrow(&mut env, "invalid error_strategy");
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
    runtime().block_on(eng.hooks.add(hook_st, config));
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineRegisterGuardrail(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    callback: JObject,
    stage: JString,
    name: JString,
) {
    let jvm = ensure_jvm(&mut env);
    let eng = match get_engine(handle as u64) {
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
            jthrow(&mut env, "invalid guardrail stage");
            return;
        }
    };
    let name_s = jstring_to_rust(&mut env, &name).unwrap_or_else(|_| "guardrail".to_string());
    let name_c = if name_s.is_empty() {
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
    let gds = Arc::clone(&eng.guardrails);
    let n = name_c.clone();
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
                name: n.clone(),
            }) as Arc<dyn GuardrailHandler>;
            let cfg = GuardrailConfig {
                name: n.clone(),
                handler: kg,
            };
            match s {
                GuardrailStage::Input => gds.add_input(cfg).await,
                GuardrailStage::Output => gds.add_output(cfg).await,
            }
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineRun(
    mut env: JNIEnv,
    _cl: JClass,
    handle: jlong,
    user_message: JString,
    timeout_secs: i64,
    connect_timeout_secs: i64,
    request_id: JString,
) -> jstring {
    let eng = match get_engine(handle as u64) {
        Ok(s) => s,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let user_message = jstring_to_rust(&mut env, &user_message).unwrap_or_default();
    let http = match effective_http(
        &eng.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let reg = Arc::clone(&eng.registry);
    let hooks = Arc::clone(&eng.hooks);
    let gds = Arc::clone(&eng.guardrails);
    let engine = SgAgentEngine::new(eng.spec.clone())
        .with_hooks(hooks)
        .with_guardrails(gds);
    let mut opts = eng.base_options.clone();
    opts.request_id = opt_request_id(&mut env, &request_id);
    let outcome = match runtime().block_on(engine.run(&http, &reg, user_message, &opts)) {
        Ok(o) => o,
        Err(e) => {
            jthrow(&mut env, &e.to_string());
            return std::ptr::null_mut();
        }
    };
    jstr_from_str(&mut env, &completion_outcome_json(outcome))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_agentEngineStream(
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
    let eng = match get_engine(handle as u64) {
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
        &eng.http,
        opt_i64(timeout_secs),
        opt_i64(connect_timeout_secs),
    ) {
        Ok(h) => h,
        Err(e) => {
            jthrow(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    let mut opts = eng.base_options.clone();
    opts.request_id = opt_request_id(&mut env, &request_id);
    let hooks = Arc::clone(&eng.hooks);
    let gds = Arc::clone(&eng.guardrails);
    let spec = eng.spec.clone();
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
    let engine = SgAgentEngine::new(spec)
        .with_hooks(hooks)
        .with_guardrails(gds);
    let stream_result = runtime().block_on(engine.stream(&http, user_message, &opts, on_delta));
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
