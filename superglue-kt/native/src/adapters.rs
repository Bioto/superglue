//! Bridge Kotlin callback interfaces to superglue `Tool` / `HookHandler` / `GuardrailHandler`.

use std::sync::Arc;

use async_trait::async_trait;
use jni::JNIEnv;
use jni::JavaVM;
use jni::objects::GlobalRef;
use jni::objects::JString;
use jni::objects::JValue;
use serde_json::{Value, json};

use superglue::guardrails::{GuardrailHandler, GuardrailOutcome, GuardrailStage};
use superglue::hooks::{HookContext, HookError, HookHandler, HookStage};
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

// ---------------------------------------------------------------------------
// JSON callback: `String invoke(String argsJson);`
// ---------------------------------------------------------------------------

pub struct KotlinJsonTool {
    pub spec: ToolSpec,
    pub jvm: Arc<JavaVM>,
    pub callback: GlobalRef,
}

pub fn call_json_callback(
    env: &mut JNIEnv,
    callback: &GlobalRef,
    args_json: &str,
) -> Result<String, String> {
    let j_arg = env
        .new_string(args_json)
        .map_err(|e| format!("new_string: {e}"))?;
    let out = env
        .call_method(
            callback,
            "invoke",
            "(Ljava/lang/String;)Ljava/lang/String;",
            &[JValue::Object(&j_arg)],
        )
        .map_err(|e| format!("call invoke: {e}"))?;
    let j_obj = out.l().map_err(|e| format!("jvalue: {e}"))?;
    if j_obj.is_null() {
        return Err("callback returned null".to_string());
    }
    let js = JString::from(j_obj);
    let java_str = env.get_string(&js).map_err(|e| e.to_string())?;
    Ok(String::from(java_str))
}

/// Same as N-API `JsonCallback` for hooks.
pub struct KotlinHook {
    pub jvm: Arc<JavaVM>,
    pub callback: GlobalRef,
    pub name: String,
}

/// `String invoke(String stage, String content)`; null = allow unchanged.
pub struct KotlinGuardrail {
    pub jvm: Arc<JavaVM>,
    pub callback: GlobalRef,
    pub stage_filter: GuardrailStateFilter,
    pub name: String,
}

pub enum GuardrailStateFilter {
    Input,
    Output,
}

#[async_trait]
impl Tool for KotlinJsonTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> std::result::Result<Value, ToolInvokeError> {
        let args_str = serde_json::to_string(&arguments)
            .map_err(|e| ToolInvokeError::handler(e.to_string(), None))?;
        let jvm = Arc::clone(&self.jvm);
        let cb = self.callback.clone();
        tokio::task::spawn_blocking(move || {
            let mut env = jvm
                .attach_current_thread()
                .map_err(|e| ToolInvokeError::handler(format!("attach: {e}"), None))?;
            call_json_callback(&mut env, &cb, &args_str)
                .map_err(|e| ToolInvokeError::handler(e, None))
                .and_then(|s| {
                    serde_json::from_str(&s)
                        .map_err(|e| ToolInvokeError::handler(e.to_string(), None))
                })
        })
        .await
        .map_err(|e| ToolInvokeError::handler(format!("join: {e}"), None))?
    }
}

#[async_trait]
impl HookHandler for KotlinHook {
    async fn execute(&self, ctx: HookContext) -> std::result::Result<HookContext, HookError> {
        let stage_str = match ctx.stage {
            HookStage::PreCompletion => "pre_completion",
            HookStage::PostCompletion => "post_completion",
            HookStage::PreTool => "pre_tool",
            HookStage::PostTool => "post_tool",
            HookStage::OnRetry => "on_retry",
            HookStage::PreBatchItem => "pre_batch_item",
            HookStage::PostBatchItem => "post_batch_item",
        };
        let meta = serde_json::to_value(&ctx.metadata)
            .map_err(|e| HookError::new(&self.name, e.to_string()))?;
        let ctx_obj = json!({
            "stage": stage_str,
            "content": ctx.content,
            "metadata": meta,
        });
        let json_in = ctx_obj.to_string();
        let jvm = Arc::clone(&self.jvm);
        let cb = self.callback.clone();
        let n = self.name.clone();
        let name_in = n.clone();
        let stage = ctx.stage.clone();
        let metadata = ctx.metadata.clone();
        let content0 = ctx.content;
        let r = match tokio::task::spawn_blocking(move || {
            let mut env = jvm
                .attach_current_thread()
                .map_err(|e| HookError::new(&name_in, format!("attach: {e}")))?;
            call_json_callback(&mut env, &cb, &json_in).map_err(|e| HookError::new(&name_in, e))
        })
        .await
        {
            Ok(x) => x,
            Err(e) => {
                return Err(HookError::new(&n, format!("join: {e}")));
            }
        };
        let result_str: String = r?;
        if result_str.is_empty() {
            return Ok(HookContext {
                stage,
                content: content0,
                metadata,
            });
        }
        let v: Value = serde_json::from_str(&result_str)
            .map_err(|e| HookError::new(&self.name, e.to_string()))?;
        let new_content: Option<String> = if v.is_null() {
            None
        } else if let Some(s) = v.as_str() {
            Some(s.to_string())
        } else if let Some(obj) = v.as_object() {
            obj.get("content")
                .and_then(|c| c.as_str())
                .map(String::from)
        } else {
            None
        };
        let out_content = new_content.unwrap_or(content0);
        Ok(HookContext {
            stage,
            content: out_content,
            metadata,
        })
    }
}

#[async_trait]
impl GuardrailHandler for KotlinGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let applies = match (&self.stage_filter, &stage) {
            (GuardrailStateFilter::Input, GuardrailStage::Input) => true,
            (GuardrailStateFilter::Output, GuardrailStage::Output) => true,
            _ => false,
        };
        if !applies {
            return GuardrailOutcome::Allow(content.to_string());
        }
        let stage_s = match stage {
            GuardrailStage::Input => "input",
            GuardrailStage::Output => "output",
        };
        let content_owned = content.to_string();
        let content_allow = content_owned.clone();
        let jvm = Arc::clone(&self.jvm);
        let cb = self.callback.clone();
        let name = self.name.clone();
        let res: Result<String, String> = tokio::task::spawn_blocking(move || {
            let mut env = jvm
                .attach_current_thread()
                .map_err(|e| format!("attach: {e}"))?;
            let s1 = env.new_string(stage_s).map_err(|e| e.to_string())?;
            let s2 = env.new_string(&content_owned).map_err(|e| e.to_string())?;
            let out = env
                .call_method(
                    &cb,
                    "invoke",
                    "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
                    &[JValue::Object(&s1), JValue::Object(&s2)],
                )
                .map_err(|e| e.to_string())?;
            let j_obj = out.l().map_err(|e| e.to_string())?;
            if j_obj.is_null() {
                return Ok(String::new());
            }
            let js = JString::from(j_obj);
            let jstr = env.get_string(&js).map_err(|e| e.to_string())?;
            Ok(String::from(jstr))
        })
        .await
        .unwrap_or_else(|e| Err(format!("join: {e}")));

        match res {
            Ok(s) if s.is_empty() => GuardrailOutcome::Allow(content_allow),
            Ok(s) => GuardrailOutcome::Allow(s),
            Err(e) => GuardrailOutcome::Block(format!("{name}: {e}")),
        }
    }
}
