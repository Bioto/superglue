//! Programmatic tool calling — the model writes JavaScript that invokes routed tools.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::Duration;

use boa_engine::{
    Context, JsError, JsNativeError, JsResult, JsValue, NativeFunction, Source, js_string,
};
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use crate::cancel::CancellationToken;

use super::error::ToolInvokeError;
use super::registry::ToolRegistry;
use super::router::{CODE_TOOL_NAME, ROUTER_TOOL_NAME};

pub const DEFAULT_CODE_TIMEOUT: Duration = Duration::from_secs(30);
pub const DEFAULT_MAX_NESTED_CALLS: u32 = 32;
pub const DEFAULT_MAX_LOOP_ITERATIONS: u64 = 1_000_000;
const MAX_SOURCE_CHARS: usize = 64_000;

#[derive(Debug, Clone)]
pub struct CodeLimits {
    pub timeout: Duration,
    pub max_nested_calls: u32,
    pub max_loop_iterations: u64,
}

impl Default for CodeLimits {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_CODE_TIMEOUT,
            max_nested_calls: DEFAULT_MAX_NESTED_CALLS,
            max_loop_iterations: DEFAULT_MAX_LOOP_ITERATIONS,
        }
    }
}

struct InvokeReq {
    name: String,
    args: Value,
    reply: std_mpsc::Sender<Result<Value, String>>,
}

/// Run `source` in an isolated JS context. Host function `tools.<name>(args)`
/// invokes allowlisted registry tools on the Tokio runtime.
pub async fn execute_code(
    source: &str,
    allowlist: &HashSet<String>,
    registry: &ToolRegistry,
    limits: CodeLimits,
) -> Result<Value, ToolInvokeError> {
    execute_code_with_cancel(source, allowlist, registry, limits, None).await
}

/// Run code while honoring an optional parent cancellation token.
pub async fn execute_code_with_cancel(
    source: &str,
    allowlist: &HashSet<String>,
    registry: &ToolRegistry,
    limits: CodeLimits,
    cancel: Option<&CancellationToken>,
) -> Result<Value, ToolInvokeError> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Err(ToolInvokeError::handler(
            "`source` must not be empty",
            Some("empty_source".into()),
        ));
    }
    if trimmed.chars().count() > MAX_SOURCE_CHARS {
        return Err(ToolInvokeError::handler(
            format!("`source` exceeds {MAX_SOURCE_CHARS} characters"),
            Some("source_too_large".into()),
        ));
    }

    let (req_tx, mut req_rx) = mpsc::unbounded_channel::<InvokeReq>();
    let (done_tx, done_rx) = oneshot::channel::<Result<Value, String>>();
    let source = trimmed.to_string();
    let max_calls = limits.max_nested_calls;
    let max_loop_iterations = limits.max_loop_iterations;
    let stop_worker = Arc::new(AtomicBool::new(false));
    let stop_worker_for_thread = Arc::clone(&stop_worker);

    tokio::task::spawn_blocking(move || {
        let result = run_js_blocking(
            &source,
            max_calls,
            max_loop_iterations,
            req_tx,
            stop_worker_for_thread,
        );
        let _ = done_tx.send(result);
    });

    let timeout = tokio::time::sleep(limits.timeout);
    tokio::pin!(timeout);
    let cancellation = async {
        match cancel {
            Some(token) => token.cancelled().await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(cancellation);
    let mut done_rx = done_rx;
    let mut calls: Vec<Value> = Vec::new();

    loop {
        tokio::select! {
            biased;
            Some(req) = req_rx.recv() => {
                let InvokeReq { name, args, reply } = req;
                if stop_worker.load(Ordering::Acquire) {
                    let _ = reply.send(Err("code execution cancelled".into()));
                    continue;
                }
                if !tool_allowed(allowlist, &name) {
                    let _ = reply.send(Err(format!(
                        "tool `{}` is not available to code",
                        name
                    )));
                    continue;
                }
                let result = tokio::select! {
                    result = registry.invoke(&name, args) => result,
                    () = &mut timeout => {
                        stop_worker.store(true, Ordering::Release);
                        return Err(ToolInvokeError::handler(
                            "code execution timed out",
                            Some("timeout".into()),
                        ));
                    }
                    () = &mut cancellation => {
                        stop_worker.store(true, Ordering::Release);
                        return Err(ToolInvokeError::handler(
                            "code execution cancelled",
                            Some("cancelled".into()),
                        ));
                    }
                };
                match result {
                    Ok(value) => {
                        calls.push(json!({"tool": name, "ok": true}));
                        let _ = reply.send(Ok(value));
                    }
                    Err(err) => {
                        calls.push(json!({
                            "tool": name,
                            "ok": false,
                            "error": err.to_string()
                        }));
                        let _ = reply.send(Err(err.to_string()));
                    }
                }
            }
            result = &mut done_rx => {
                return finish_code(result, calls);
            }
            () = &mut timeout => {
                stop_worker.store(true, Ordering::Release);
                return Err(ToolInvokeError::handler(
                    "code execution timed out",
                    Some("timeout".into()),
                ));
            }
            () = &mut cancellation => {
                stop_worker.store(true, Ordering::Release);
                return Err(ToolInvokeError::handler(
                    "code execution cancelled",
                    Some("cancelled".into()),
                ));
            }
        }
    }
}

fn tool_allowed(allowlist: &HashSet<String>, name: &str) -> bool {
    if name == CODE_TOOL_NAME || name == ROUTER_TOOL_NAME {
        return false;
    }
    allowlist.contains(name)
}

fn finish_code(
    result: Result<Result<Value, String>, oneshot::error::RecvError>,
    calls: Vec<Value>,
) -> Result<Value, ToolInvokeError> {
    match result {
        Ok(Ok(value)) => Ok(json!({
            "ok": true,
            "result": value,
            "calls": calls,
        })),
        Ok(Err(message)) => Ok(json!({
            "ok": false,
            "error": message,
            "calls": calls,
        })),
        Err(_) => Err(ToolInvokeError::handler(
            "code execution worker ended unexpectedly",
            Some("worker".into()),
        )),
    }
}

fn run_js_blocking(
    source: &str,
    max_calls: u32,
    max_loop_iterations: u64,
    req_tx: mpsc::UnboundedSender<InvokeReq>,
    stop_worker: Arc<AtomicBool>,
) -> Result<Value, String> {
    let mut context = Context::default();
    context
        .runtime_limits_mut()
        .set_loop_iteration_limit(max_loop_iterations);
    let call_count = Arc::new(AtomicU32::new(0));
    let req_tx = Arc::new(req_tx);
    let call_count_host = Arc::clone(&call_count);
    let req_tx_host = Arc::clone(&req_tx);
    let stop_worker_host = Arc::clone(&stop_worker);

    // Safety: the closure only holds `Arc`s (no JS values). It does not outlive
    // this `Context`.
    let invoke = unsafe {
        NativeFunction::from_closure(move |_this, args, context| -> JsResult<JsValue> {
            invoke_from_js(
                args,
                context,
                &req_tx_host,
                &call_count_host,
                max_calls,
                &stop_worker_host,
            )
        })
    };
    context
        .register_global_callable(js_string!("__invoke"), 2, invoke)
        .map_err(|e| e.to_string())?;

    let wrapped = format!(
        "(function() {{\n\
           const tools = new Proxy({{}}, {{\n\
             get(_target, prop) {{\n\
               if (prop === 'call') {{\n\
                 return (name, args) => __invoke(String(name), args ?? {{}});\n\
               }}\n\
               return (args) => __invoke(String(prop), args ?? {{}});\n\
             }}\n\
           }});\n\
           {source}\n\
         }})()"
    );

    let value = context
        .eval(Source::from_bytes(wrapped.as_bytes()))
        .map_err(|e| e.to_string())?;
    js_to_json(&value, &mut context)
}

fn invoke_from_js(
    args: &[JsValue],
    context: &mut Context,
    req_tx: &mpsc::UnboundedSender<InvokeReq>,
    call_count: &AtomicU32,
    max_calls: u32,
    stop_worker: &AtomicBool,
) -> JsResult<JsValue> {
    if stop_worker.load(Ordering::Acquire) {
        return Err(JsNativeError::error()
            .with_message("code execution cancelled")
            .into());
    }
    let name = args
        .first()
        .cloned()
        .unwrap_or(JsValue::undefined())
        .to_string(context)?
        .to_std_string_escaped();
    if name == CODE_TOOL_NAME || name == ROUTER_TOOL_NAME {
        return Err(JsNativeError::typ()
            .with_message(format!("tool `{name}` cannot be called from code"))
            .into());
    }
    let prev = call_count.fetch_add(1, Ordering::SeqCst);
    if prev >= max_calls {
        return Err(JsNativeError::range()
            .with_message(format!("exceeded max nested tool calls ({max_calls})"))
            .into());
    }
    let args_json = match args.get(1) {
        Some(value) if !value.is_undefined() && !value.is_null() => js_to_json(value, context)
            .map_err(|e| JsError::from(JsNativeError::typ().with_message(e)))?,
        _ => json!({}),
    };

    let (reply_tx, reply_rx) = std_mpsc::channel();
    req_tx
        .send(InvokeReq {
            name,
            args: args_json,
            reply: reply_tx,
        })
        .map_err(|_| JsNativeError::error().with_message("code runtime closed"))?;
    loop {
        match reply_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(Ok(value)) => return json_to_js(&value, context),
            Ok(Err(message)) => {
                return Err(JsNativeError::error().with_message(message).into());
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {
                if stop_worker.load(Ordering::Acquire) {
                    return Err(JsNativeError::error()
                        .with_message("code execution cancelled")
                        .into());
                }
            }
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                return Err(JsNativeError::error()
                    .with_message("tool invoke cancelled")
                    .into());
            }
        }
    }
}

fn js_to_json(value: &JsValue, context: &mut Context) -> Result<Value, String> {
    if value.is_undefined() {
        return Ok(Value::Null);
    }
    value
        .to_json(context)
        .map_err(|e| e.to_string())
        .and_then(|opt| opt.ok_or_else(|| "could not convert JS value to JSON".into()))
}

fn json_to_js(value: &Value, context: &mut Context) -> JsResult<JsValue> {
    JsValue::from_json(value, context)
}

/// Parse the `source` argument from a `code` tool call.
pub fn source_from_arguments(arguments: &Value) -> Result<String, ToolInvokeError> {
    arguments
        .get("source")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            ToolInvokeError::handler(
                "`code` requires a non-empty `source` string",
                Some("missing_source".into()),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Tool, ToolSpec};
    use async_trait::async_trait;

    struct AddTool;

    #[async_trait]
    impl Tool for AddTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "add".into(),
                description: Some("add two numbers".into()),
                parameters_schema: json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }),
                static_tool: false,
            }
        }

        async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
            let a = arguments.get("a").and_then(Value::as_i64).unwrap_or(0);
            let b = arguments.get("b").and_then(Value::as_i64).unwrap_or(0);
            Ok(json!({"sum": a + b}))
        }
    }

    fn allow(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    #[tokio::test]
    async fn script_calls_allowlisted_tool_and_reduces() {
        let registry = ToolRegistry::new();
        registry
            .register(std::sync::Arc::new(AddTool))
            .await
            .unwrap();
        let out = execute_code(
            r#"
              const a = tools.add({a: 2, b: 3});
              const b = tools.call("add", {a: a.sum, b: 4});
              return { total: b.sum };
            "#,
            &allow(&["add"]),
            &registry,
            CodeLimits::default(),
        )
        .await
        .unwrap();
        assert_eq!(out["ok"], true);
        assert_eq!(out["result"]["total"], 9);
        assert_eq!(out["calls"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error_result() {
        let registry = ToolRegistry::new();
        let out = execute_code(
            r#"return tools.missing({});"#,
            &allow(&["add"]),
            &registry,
            CodeLimits::default(),
        )
        .await
        .unwrap();
        assert_eq!(out["ok"], false);
        assert!(
            out["error"]
                .as_str()
                .unwrap()
                .contains("not available to code")
        );
    }

    #[tokio::test]
    async fn cannot_call_code_or_router_from_script() {
        let registry = ToolRegistry::new();
        let out = execute_code(
            r#"return tools.code({source: "return 1"});"#,
            &allow(&[CODE_TOOL_NAME]),
            &registry,
            CodeLimits::default(),
        )
        .await
        .unwrap();
        assert_eq!(out["ok"], false);
        assert!(out["error"].as_str().unwrap().contains("cannot be called"));
    }

    #[tokio::test]
    async fn empty_source_is_rejected() {
        let registry = ToolRegistry::new();
        let err = execute_code("   ", &allow(&[]), &registry, CodeLimits::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn tight_loop_returns_without_pinning_worker() {
        let registry = ToolRegistry::new();
        let limits = CodeLimits {
            timeout: Duration::from_millis(10),
            max_loop_iterations: 1_000_000,
            ..CodeLimits::default()
        };

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            execute_code("while (true) {}", &allow(&[]), &registry, limits),
        )
        .await
        .expect("tight loop should return promptly");

        match result {
            Ok(value) => {
                assert_eq!(value["ok"], false);
                assert!(value["error"].as_str().unwrap().contains("loop"));
            }
            Err(err) => assert!(err.to_string().contains("timed out")),
        }
    }

    #[tokio::test]
    async fn pre_cancelled_execution_stops_before_running_code() {
        let registry = ToolRegistry::new();
        let token = CancellationToken::new();
        token.cancel();

        let err = execute_code_with_cancel(
            "while (true) {}",
            &allow(&[]),
            &registry,
            CodeLimits::default(),
            Some(&token),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("cancelled"));
    }
}
