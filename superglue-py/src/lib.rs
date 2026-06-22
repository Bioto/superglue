//! Python extension: superglue
//!
//! Rust-side [`tracing`] output is initialised when this native module loads (idempotent).
//! Set the `RUST_LOG` environment variable before importing (for example `RUST_LOG=info` or
//! `RUST_LOG=superglue=debug`) to control log verbosity; see the `tracing-subscriber`
//! `EnvFilter` documentation for filter syntax.
//!
//! # Quick-start
//!
//! ```python
//! import superglue
//!
//! client = superglue.Client(
//!     api_key="sk-...",
//!     model="gpt-5.4-nano-2026-03-17-mini",
//!     system_prompt="You are a helpful assistant.",
//! )
//!
//! def get_weather(args: dict) -> dict:
//!     return {"temp": 22, "unit": "C"}
//!
//! client.register_tool(
//!     name="get_weather",
//!     description="Get current weather for a location.",
//!     parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
//!     fn=get_weather,
//! )
//!
//! result = client.complete("What is the weather in Paris?")
//! print(result.content)
//! print(result.usage)
//! ```

use std::sync::{Arc, Mutex, OnceLock};

use secrecy::Secret;

use async_trait::async_trait;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use tokio::runtime::Runtime;

use std::num::NonZeroU32;
use std::collections::HashMap;
use std::time::Duration;

use superglue::agents::{AgentEngine, AgentSpec};
use superglue::batch::{BatchConfig, BatchRequest, ErrorStrategy, batch_complete};
use superglue::chat::{
    ChatOptions, CompletionOutcome, StreamOutcome, complete_with_tools, stream_complete,
    stream_complete_with_tools,
};
use superglue::fallback::ModelFallbackChain;
use superglue::responses::{
    ResponseOutcome, ResponseStreamOutcome, complete_with_tools as complete_response_with_tools,
    stream_response as stream_response_api,
};
use superglue::mcp::{McpHttpConfig, McpSession, McpStdioConfig};
use superglue::guardrails::{
    BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailError, GuardrailHandler,
    GuardrailOutcome, GuardrailRegistry, GuardrailStage, LengthStrategy, MaxLengthGuardrail,
    PiiRedactGuardrail,
};
use superglue::hooks::{
    HookConfig, HookContext, HookError, HookErrorStrategy, HookHandler, HookRegistry, HookStage,
};
use superglue::client::{
    bootstrap_from_parts, provider_id_from_str, BindingBootstrapConfig, ClientBuildError,
};
use superglue::files::{self, FilePurpose};
use superglue::http::{ClientConfig, HttpClient, RetryPolicy};
use superglue::openai::ChatMessage;
use superglue::telemetry::{TelemetryConfig, init_tracing};
use superglue::events::{
    ProcessEvent, ProcessEventKind, StatusEmitter, StatusSubscriber,
};
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| Runtime::new().expect("tokio runtime"))
}

// ---------------------------------------------------------------------------
// Tool adapter: Python callable receiving/returning dicts
// ---------------------------------------------------------------------------

/// Wraps a Python `def fn(args: dict) -> dict` as a Rust [`Tool`].
///
/// The binding handles JSON serialization so callers never touch JSON strings.
struct PythonDictTool {
    spec: ToolSpec,
    callback: Arc<Py<PyAny>>,
}

#[async_trait]
impl Tool for PythonDictTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let args_json = serde_json::to_string(&arguments)
            .map_err(|e| ToolInvokeError::handler(e.to_string(), None))?;
        let cb = Arc::clone(&self.callback);
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                // Convert JSON → Python dict via json.loads
                let json_mod = py
                    .import("json")
                    .map_err(|e| ToolInvokeError::handler(format!("import json: {e}"), None))?;
                let args_dict = json_mod
                    .call_method1("loads", (args_json.as_str(),))
                    .map_err(|e| ToolInvokeError::handler(format!("json.loads: {e}"), None))?;

                // Call the Python function with the dict
                let result = cb
                    .call1(py, (args_dict,))
                    .map_err(|e| ToolInvokeError::handler(format!("python call: {e}"), None))?;

                // Convert return value → JSON string via json.dumps
                let result_str: String = json_mod
                    .call_method1("dumps", (result,))
                    .map_err(|e| ToolInvokeError::handler(format!("json.dumps: {e}"), None))?
                    .extract()
                    .map_err(|e| ToolInvokeError::handler(format!("extract str: {e}"), None))?;

                serde_json::from_str(&result_str)
                    .map_err(|e| ToolInvokeError::handler(e.to_string(), None))
            })
        })
        .await
        .map_err(|e| ToolInvokeError::handler(format!("task join: {e}"), None))?
    }
}

fn tool_error_to_py(e: ToolInvokeError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Guardrail adapter: Python callable receiving content string, returning str or None
// ---------------------------------------------------------------------------

/// Which pipeline side a PythonGuardrail applies to.
#[derive(Clone)]
enum PyGuardrailStageFilter {
    Input,
    Output,
    Both,
}

/// Wraps a Python callable as a Rust [`GuardrailHandler`].
///
/// The Python function receives `(content: str) -> str | None`:
/// - Return `None` or the same string → allow unchanged.
/// - Return a different string → allow with transformation.
/// - Raise any exception → block (the exception message becomes the reason).
struct PythonGuardrail {
    callback: Arc<Py<PyAny>>,
    stage_filter: PyGuardrailStageFilter,
    name: String,
}

#[async_trait]
impl GuardrailHandler for PythonGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let applies = match (&self.stage_filter, &stage) {
            (PyGuardrailStageFilter::Both, _) => true,
            (PyGuardrailStageFilter::Input, GuardrailStage::Input) => true,
            (PyGuardrailStageFilter::Output, GuardrailStage::Output) => true,
            _ => false,
        };
        if !applies {
            return GuardrailOutcome::Allow(content.to_string());
        }

        let cb = Arc::clone(&self.callback);
        let content_owned = content.to_string();
        let name = self.name.clone();

        let stage_str = match stage {
            GuardrailStage::Input => "input",
            GuardrailStage::Output => "output",
        };

        tokio::task::spawn_blocking(move || {
            Python::attach(
                |py| match cb.call1(py, (stage_str, content_owned.as_str())) {
                    Ok(result) => {
                        if result.is_none(py) {
                            GuardrailOutcome::Allow(content_owned)
                        } else if let Ok(s) = result.extract::<String>(py) {
                            GuardrailOutcome::Allow(s)
                        } else {
                            GuardrailOutcome::Allow(content_owned)
                        }
                    }
                    Err(e) => GuardrailOutcome::Block(format!("{}: {}", name, e)),
                },
            )
        })
        .await
        .unwrap_or_else(|e| GuardrailOutcome::Block(format!("spawn_blocking error: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Hook adapter: Python callable receiving/returning a context dict
// ---------------------------------------------------------------------------

/// Wraps a Python callable as a Rust [`HookHandler`].
///
/// The Python function receives a dict with keys:
///   - `stage` (str): hook stage name
///   - `content` (str): stage-specific payload
///   - `metadata` (dict): additional key-value pairs
///
/// It should return either:
///   - `None` / `str`: replaces `content` (None = no change)
///   - A dict with a `"content"` key
struct PythonHook {
    callback: Arc<Py<PyAny>>,
    name: String,
}

#[async_trait]
impl HookHandler for PythonHook {
    async fn execute(&self, ctx: HookContext) -> Result<HookContext, HookError> {
        let cb = Arc::clone(&self.callback);
        let name = self.name.clone();

        tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                let json_mod = py
                    .import("json")
                    .map_err(|e| HookError::new(&name, format!("import json: {e}")))?;

                // Build metadata dict
                let meta_str = serde_json::to_string(&ctx.metadata)
                    .map_err(|e| HookError::new(&name, e.to_string()))?;
                let meta_dict = json_mod
                    .call_method1("loads", (meta_str.as_str(),))
                    .map_err(|e| HookError::new(&name, format!("json.loads metadata: {e}")))?;

                // Build the stage name string
                let stage_str = match ctx.stage {
                    HookStage::PreCompletion => "pre_completion",
                    HookStage::PostCompletion => "post_completion",
                    HookStage::PreTool => "pre_tool",
                    HookStage::PostTool => "post_tool",
                    HookStage::OnRetry => "on_retry",
                    HookStage::PreBatchItem => "pre_batch_item",
                    HookStage::PostBatchItem => "post_batch_item",
                };

                // Assemble context dict: {"stage": str, "content": str, "metadata": dict}
                let ctx_dict = pyo3::types::PyDict::new(py);
                ctx_dict
                    .set_item("stage", stage_str)
                    .map_err(|e| HookError::new(&name, e.to_string()))?;
                ctx_dict
                    .set_item("content", ctx.content.as_str())
                    .map_err(|e| HookError::new(&name, e.to_string()))?;
                ctx_dict
                    .set_item("metadata", meta_dict)
                    .map_err(|e| HookError::new(&name, e.to_string()))?;

                // Call the Python function
                let result = cb
                    .call1(py, (ctx_dict,))
                    .map_err(|e| HookError::new(&name, format!("python call: {e}")))?;

                // Interpret the return value
                let new_content: Option<String> = if result.is_none(py) {
                    None
                } else if let Ok(s) = result.extract::<String>(py) {
                    Some(s)
                } else if let Ok(d) = result.downcast_bound::<pyo3::types::PyDict>(py) {
                    d.get_item("content")
                        .ok()
                        .flatten()
                        .and_then(|v| v.extract::<String>().ok())
                } else {
                    None
                };

                let content = new_content.unwrap_or(ctx.content);
                Ok(HookContext {
                    stage: ctx.stage,
                    content,
                    metadata: ctx.metadata,
                })
            })
        })
        .await
        .map_err(|e| HookError::new("spawn_blocking", e.to_string()))?
    }
}

/// Return the caller's `Arc<HttpClient>` unchanged, or build a derived one with
/// overridden timeouts when either `timeout_secs` / `connect_timeout_secs` is set.
fn py_to_chat_messages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ChatMessage>> {
    let py = obj.py();
    let json = py.import("json")?;
    let dumped: String = json.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&dumped).map_err(|e| {
        PyValueError::new_err(format!(
            "messages must be a sequence of JSON-serializable OpenAI-style message dicts: {e}"
        ))
    })
}

fn chat_messages_to_py_any(py: Python<'_>, msgs: &[ChatMessage]) -> PyResult<Py<PyAny>> {
    let s = serde_json::to_string(msgs).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let json = py.import("json")?;
    let obj = json.call_method1("loads", (s.as_str(),))?;
    Ok(obj.unbind().into_any())
}

fn completion_outcome_to_py(
    py: Python<'_>,
    outcome: CompletionOutcome,
) -> PyResult<PyCompletionOutcome> {
    let usage_py = outcome.usage.map(|u| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("prompt_tokens", u.prompt_tokens).ok();
        dict.set_item("completion_tokens", u.completion_tokens).ok();
        dict.set_item("total_tokens", u.total_tokens).ok();
        dict.unbind().into_any()
    });
    let messages_py = chat_messages_to_py_any(py, &outcome.messages)?;
    Ok(PyCompletionOutcome {
        content: outcome.content,
        rounds: outcome.rounds,
        usage: usage_py,
        request_id: outcome.request_id,
        model_used: outcome.model_used,
        messages: messages_py,
    })
}

fn effective_http(
    base: &Arc<HttpClient>,
    timeout_secs: Option<u64>,
    connect_timeout_secs: Option<u64>,
) -> PyResult<Arc<HttpClient>> {
    if timeout_secs.is_none() && connect_timeout_secs.is_none() {
        return Ok(Arc::clone(base));
    }
    let t = timeout_secs
        .map(Duration::from_secs)
        .unwrap_or(base.config.timeout);
    let ct = connect_timeout_secs
        .map(Duration::from_secs)
        .unwrap_or(base.config.connect_timeout);
    base.clone_with_timeouts(t, ct)
        .map(Arc::new)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// ---------------------------------------------------------------------------
// PyCompletionOutcome
// ---------------------------------------------------------------------------

/// Result returned by [`Client.complete`].
///
/// Attributes:
///   content (str | None): final assistant text.
///   rounds (int): number of HTTP completion calls made.
///   usage (dict | None): token usage {"prompt_tokens", "completion_tokens", "total_tokens"}.
///   request_id (str): correlation ID (UUID v4 auto-generated or caller-supplied).
///   messages (list[dict]): caller-visible chat history after this turn (no synthetic system row).
#[pyclass(name = "CompletionOutcome")]
struct PyCompletionOutcome {
    #[pyo3(get)]
    content: Option<String>,
    #[pyo3(get)]
    rounds: u32,
    /// Token usage as a Python dict, or None when the provider didn't return it.
    #[pyo3(get)]
    usage: Option<Py<PyAny>>,
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    model_used: String,
    #[pyo3(get)]
    messages: Py<PyAny>,
}

#[pymethods]
impl PyCompletionOutcome {
    fn __repr__(&self) -> String {
        format!(
            "CompletionOutcome(content={:?}, rounds={}, usage={:?}, request_id={:?}, messages=<list>)",
            self.content,
            self.rounds,
            self.usage.as_ref().map(|_| "<dict>"),
            self.request_id,
        )
    }
}

// ---------------------------------------------------------------------------
// PyStreamOutcome
// ---------------------------------------------------------------------------

/// Result returned by [`Client.stream`].
///
/// Attributes:
///   content (str): full accumulated assistant text.
///   finish_reason (str | None): why the stream stopped (e.g. "stop", "length").
///   usage (dict | None): token usage when `include_usage` was enabled.
///   request_id (str): correlation ID (UUID v4 auto-generated or caller-supplied).
#[pyclass(name = "StreamOutcome")]
struct PyStreamOutcome {
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    finish_reason: Option<String>,
    #[pyo3(get)]
    usage: Option<Py<PyAny>>,
    #[pyo3(get)]
    request_id: String,
}

#[pymethods]
impl PyStreamOutcome {
    fn __repr__(&self) -> String {
        format!(
            "StreamOutcome(content={:?}, finish_reason={:?}, usage={}, request_id={:?})",
            self.content,
            self.finish_reason,
            if self.usage.is_some() {
                "<dict>"
            } else {
                "None"
            },
            self.request_id,
        )
    }
}

fn usage_to_py(py: Python<'_>, u: &superglue::proto::Usage) -> Py<PyAny> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("prompt_tokens", u.prompt_tokens).ok();
    dict.set_item("completion_tokens", u.completion_tokens).ok();
    dict.set_item("total_tokens", u.total_tokens).ok();
    dict.unbind().into_any()
}

/// Result returned by [`Client.complete_response`] and [`Client.stream_response`].
#[pyclass(name = "ResponseOutcome")]
struct PyResponseOutcome {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    content: Option<String>,
    #[pyo3(get)]
    rounds: u32,
    #[pyo3(get)]
    usage: Option<Py<PyAny>>,
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    model_used: String,
}

fn response_outcome_to_py(py: Python<'_>, o: ResponseOutcome) -> PyResponseOutcome {
    let usage_py = o.usage.as_ref().map(|u| usage_to_py(py, u));
    PyResponseOutcome {
        id: o.id,
        content: o.content,
        rounds: o.rounds,
        usage: usage_py,
        request_id: o.request_id,
        model_used: o.model_used,
    }
}

/// Result returned by [`Client.stream_response`].
#[pyclass(name = "ResponseStreamOutcome")]
struct PyResponseStreamOutcome {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    usage: Option<Py<PyAny>>,
    #[pyo3(get)]
    request_id: String,
    #[pyo3(get)]
    model_used: String,
}

fn response_stream_outcome_to_py(py: Python<'_>, o: ResponseStreamOutcome) -> PyResponseStreamOutcome {
    let usage_py = o.usage.as_ref().map(|u| usage_to_py(py, u));
    PyResponseStreamOutcome {
        id: o.id,
        content: o.content,
        usage: usage_py,
        request_id: o.request_id,
        model_used: o.model_used,
    }
}

fn stream_outcome_to_py(py: Python<'_>, o: StreamOutcome) -> PyStreamOutcome {
    let usage_py = o.usage.map(|u| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("prompt_tokens", u.prompt_tokens).ok();
        dict.set_item("completion_tokens", u.completion_tokens).ok();
        dict.set_item("total_tokens", u.total_tokens).ok();
        dict.unbind().into_any()
    });
    PyStreamOutcome {
        content: o.content,
        finish_reason: o.finish_reason,
        usage: usage_py,
        request_id: o.request_id,
    }
}

// ---------------------------------------------------------------------------
// StatusEmitter + ProcessEvent (Python bindings)
// ---------------------------------------------------------------------------

fn process_event_kind_str(kind: ProcessEventKind) -> &'static str {
    kind.as_str()
}

fn process_event_to_py(py: Python<'_>, event: &ProcessEvent) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("kind", process_event_kind_str(event.kind))?;
    dict.set_item("request_id", &event.request_id)?;
    dict.set_item("round", event.round)?;
    dict.set_item("model", &event.model)?;
    dict.set_item("tool_call_count", event.tool_call_count)?;
    if let Some(usage) = &event.usage {
        let u = pyo3::types::PyDict::new(py);
        u.set_item("prompt_tokens", usage.prompt_tokens)?;
        u.set_item("completion_tokens", usage.completion_tokens)?;
        u.set_item("total_tokens", usage.total_tokens)?;
        dict.set_item("usage", u)?;
    } else {
        dict.set_item("usage", py.None())?;
    }
    dict.set_item("estimated_cost_usd", event.estimated_cost_usd)?;
    dict.set_item("error_type", &event.error_type)?;
    dict.set_item("timestamp_ms", event.timestamp_ms)?;
    let meta = pyo3::types::PyDict::new(py);
    for (k, v) in &event.metadata {
        meta.set_item(k, v)?;
    }
    dict.set_item("metadata", meta)?;
    Ok(dict.into())
}

struct PythonStatusSubscriber {
    callback: Arc<Py<PyAny>>,
}

#[async_trait]
impl StatusSubscriber for PythonStatusSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        let callback = Arc::clone(&self.callback);
        let _ = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                if let Ok(dict) = process_event_to_py(py, &event) {
                    let _ = callback.call1(py, (dict,));
                }
            });
        })
        .await;
    }
}

#[pyclass(name = "_RustStatusEmitter")]
#[derive(Clone)]
struct PyStatusEmitter {
    inner: Arc<StatusEmitter>,
}

#[pymethods]
impl PyStatusEmitter {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(StatusEmitter::new()),
        }
    }

    fn subscribe(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let subscriber = Arc::new(PythonStatusSubscriber {
            callback: Arc::new(callback.clone_ref(py)),
        });
        let inner = Arc::clone(&self.inner);
        runtime()
            .block_on(inner.subscribe(subscriber as Arc<dyn StatusSubscriber>));
        Ok(())
    }
}

fn py_str_dict(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut out = HashMap::new();
    for (k, v) in dict.iter() {
        out.insert(k.extract()?, v.extract()?);
    }
    Ok(out)
}

fn provider_credentials_from_api_keys(
    api_keys: &HashMap<String, String>,
) -> superglue::providers::ProviderCredentials {
    let mut creds = superglue::providers::ProviderCredentials::from_env();
    for (name, key) in api_keys {
        if let Some(pid) = provider_id_from_str(name) {
            creds.insert_key(pid, key);
        }
    }
    creds
}

fn provider_qps_from_map(map: &HashMap<String, u32>) -> HashMap<superglue::providers::ProviderId, u32> {
    let mut out = HashMap::new();
    for (name, qps) in map {
        if let Some(pid) = provider_id_from_str(name) {
            out.insert(pid, *qps);
        }
    }
    out
}

fn file_purpose_from_str(s: &str) -> PyResult<FilePurpose> {
    match s {
        "assistants" => Ok(FilePurpose::Assistants),
        "user_data" => Ok(FilePurpose::UserData),
        "batch" => Ok(FilePurpose::Batch),
        other => Err(PyValueError::new_err(format!("unknown file purpose: {other}"))),
    }
}

fn finalize_call_options(
    base: &ChatOptions,
    status_emitter: &Option<Arc<StatusEmitter>>,
    request_id: Option<String>,
    reasoning_effort: Option<String>,
) -> ChatOptions {
    let mut options = base.clone();
    options.request_id = request_id;
    options.status_emitter = status_emitter.clone();
    if let Some(re) = reasoning_effort {
        options.reasoning_effort = Some(re);
    }
    options
}

// ---------------------------------------------------------------------------
// PyClient
// ---------------------------------------------------------------------------

/// High-level client: create once, register tools, call `complete()`.
///
/// Args:
///   api_key (str): Provider API key.
///   model (str): Model identifier (default "gpt-5.4-nano-2026-03-17-mini").
///   base_url (str): API base URL (default "https://api.openai.com").
///   system_prompt (str | None): System message prepended to every conversation.
///   max_tool_rounds (int): Maximum completion calls per turn (default 16).
///   max_retries (int): Maximum retry attempts on transient errors (default 3).
///   retry_initial_delay_ms (int): Initial backoff delay in milliseconds (default 50).
///   retry_max_delay_ms (int): Maximum backoff delay in milliseconds (default 2000).
///   retry_multiplier (float): Exponential backoff multiplier (default 2.0).
///   requests_per_second (int | None): Optional QPS cap; None means unlimited (default None).
///   timeout_secs (int): Total per-request timeout in seconds (default 60).
///   connect_timeout_secs (int): Connection timeout in seconds (default 30).
///   pool_max_idle_per_host (int): Max idle keep-alive connections per host in the
///     connection pool (default 50).
///   pool_idle_timeout_secs (int | None): Seconds before an idle connection is evicted
///     from the pool. None uses reqwest's default of 90 s (default None).
#[pyclass(name = "Client")]
struct PyClient {
    options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
    status_emitter: Option<Arc<StatusEmitter>>,
    max_upload_bytes: usize,
    mcp_sessions: Mutex<Vec<Arc<McpSession>>>,
}

#[pyclass(name = "UploadedFile")]
struct PyUploadedFile {
    #[pyo3(get)]
    provider: String,
    #[pyo3(get)]
    file_id: String,
    #[pyo3(get)]
    filename: String,
    #[pyo3(get)]
    bytes: usize,
    #[pyo3(get)]
    purpose: String,
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (
        api_key,
        model = "gpt-5.4-nano-2026-03-17-mini",
        base_url = "https://api.openai.com",
        system_prompt = None,
        max_tool_rounds = 16,
        max_retries = 3u32,
        retry_initial_delay_ms = 50u64,
        retry_max_delay_ms = 2000u64,
        retry_multiplier = 2.0f64,
        requests_per_second = None,
        timeout_secs = 60u64,
        connect_timeout_secs = 30u64,
        max_output_retries = 3u32,
        pool_max_idle_per_host = 50usize,
        pool_idle_timeout_secs = None,
        reasoning_effort = None,
        status_emitter = None,
        model_fallback_models = None,
        api_keys = None,
        requests_per_second_for = None,
        max_upload_bytes = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        api_key: String,
        model: &str,
        base_url: &str,
        system_prompt: Option<String>,
        max_tool_rounds: u32,
        max_retries: u32,
        retry_initial_delay_ms: u64,
        retry_max_delay_ms: u64,
        retry_multiplier: f64,
        requests_per_second: Option<u32>,
        timeout_secs: u64,
        connect_timeout_secs: u64,
        max_output_retries: u32,
        pool_max_idle_per_host: usize,
        pool_idle_timeout_secs: Option<u64>,
        reasoning_effort: Option<String>,
        status_emitter: Option<PyStatusEmitter>,
        model_fallback_models: Option<Vec<String>>,
        api_keys: Option<Bound<'_, PyDict>>,
        requests_per_second_for: Option<Bound<'_, PyDict>>,
        max_upload_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let api_keys_map = api_keys
            .map(|d| py_str_dict(&d))
            .transpose()?
            .unwrap_or_default();
        let provider_creds = if api_keys_map.is_empty() {
            None
        } else {
            Some(provider_credentials_from_api_keys(&api_keys_map))
        };
        let qps_for_map = if let Some(d) = requests_per_second_for {
            let m = py_str_dict(&d)?;
            let mut out: HashMap<String, u32> = HashMap::new();
            for (k, v) in m {
                let n: u32 = v.parse().map_err(|_| {
                    PyValueError::new_err(format!("invalid qps for {k}: {v}"))
                })?;
                out.insert(k, n);
            }
            out
        } else {
            HashMap::new()
        };

        let model_fallback = model_fallback_models.and_then(|models| {
            if models.is_empty() {
                None
            } else {
                Some(ModelFallbackChain::new(models))
            }
        });

        let status_emitter_arc = status_emitter.map(|e| e.inner);

        let bootstrap = bootstrap_from_parts(BindingBootstrapConfig {
            api_key,
            base_url: base_url.to_string(),
            model: model.to_string(),
            system_prompt,
            max_tool_rounds,
            max_retries,
            retry_initial_delay_ms,
            retry_max_delay_ms,
            retry_multiplier,
            requests_per_second,
            timeout: Duration::from_secs(timeout_secs),
            connect_timeout: Duration::from_secs(connect_timeout_secs),
            max_output_retries,
            pool_max_idle_per_host,
            pool_idle_timeout: pool_idle_timeout_secs.map(Duration::from_secs),
            reasoning_effort,
            status_emitter: status_emitter_arc.clone(),
            model_fallback,
            provider_credentials: provider_creds,
            provider_qps: provider_qps_from_map(&qps_for_map),
            max_upload_bytes: max_upload_bytes.unwrap_or_else(files::default_max_upload_bytes),
        })
        .map_err(|e: ClientBuildError| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            options: bootstrap.options,
            registry: Arc::new(ToolRegistry::new()),
            hooks: Arc::new(HookRegistry::new()),
            guardrails: Arc::new(
                GuardrailRegistry::new().with_max_output_retries(max_output_retries),
            ),
            http: Arc::new(bootstrap.http),
            status_emitter: status_emitter_arc,
            max_upload_bytes: bootstrap.max_upload_bytes,
            mcp_sessions: Mutex::new(Vec::new()),
        })
    }

    /// Upload a file to the provider Files API (OpenAI/xAI) or return inline metadata (Anthropic).
    #[pyo3(signature = (path, purpose = "user_data", provider = None))]
    fn upload_file(
        &self,
        path: String,
        purpose: &str,
        provider: Option<String>,
    ) -> PyResult<PyUploadedFile> {
        let purpose = file_purpose_from_str(purpose)?;
        let provider_id = provider
            .as_deref()
            .and_then(provider_id_from_str)
            .or_else(|| {
                Some(superglue::providers::parse_model_ref(&self.options.model).provider)
            })
            .expect("model provider");
        let creds = superglue::chat::credentials_for(&self.options);
        let uploaded = runtime()
            .block_on(files::upload_file(
                &self.http,
                &creds,
                provider_id,
                std::path::Path::new(&path),
                purpose,
                self.max_upload_bytes,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyUploadedFile {
            provider: uploaded.provider.to_string(),
            file_id: uploaded.file_id,
            filename: uploaded.filename,
            bytes: uploaded.bytes,
            purpose: uploaded.purpose.as_openai_str().to_string(),
        })
    }

    /// Build a user message with inline file bytes (base64-encoded for the API).
    #[staticmethod]
    #[pyo3(signature = (filename, file_bytes, text=None))]
    fn message_with_file_bytes(
        py: Python<'_>,
        filename: String,
        file_bytes: &[u8],
        text: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let msg = files::message_with_file_bytes(text.as_deref(), &filename, file_bytes);
        let s = serde_json::to_string(&msg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json = py.import("json")?;
        let obj = json.call_method1("loads", (s.as_str(),))?;
        Ok(obj.unbind().into_any())
    }

    /// Register a tool callable.
    ///
    /// Args:
    ///   name (str): Tool name (must be unique).
    ///   description (str): Human-readable description shown to the model.
    ///   parameters (dict): JSON Schema dict describing the tool's parameters.
    ///   fn (callable): Python function `def fn(args: dict) -> dict`.
    #[pyo3(signature = (name, description, parameters, r#fn))]
    fn register_tool(
        &self,
        py: Python<'_>,
        name: String,
        description: String,
        parameters: Bound<'_, PyAny>,
        r#fn: Py<PyAny>,
    ) -> PyResult<()> {
        // Convert Python dict → JSON string → serde_json::Value
        let json_mod = py.import("json")?;
        let params_json: String = json_mod.call_method1("dumps", (parameters,))?.extract()?;
        let parameters_schema: Value =
            serde_json::from_str(&params_json).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let spec = ToolSpec {
            name,
            description: Some(description),
            parameters_schema,
        };
        let callback = Arc::new(r#fn.clone_ref(py));
        let tool = Arc::new(PythonDictTool { spec, callback }) as Arc<dyn Tool>;
        runtime()
            .block_on(self.registry.register(tool))
            .map_err(tool_error_to_py)
    }

    /// Register a lifecycle hook.
    ///
    /// Args:
    ///   stage (str): One of "pre_completion", "post_completion", "pre_tool", "post_tool",
    ///                "on_retry", "pre_batch_item", "post_batch_item".
    ///   handler (callable): ``fn(ctx: dict) -> str | dict | None``
    ///     Receives ``{"stage": str, "content": str, "metadata": dict}``.
    ///     Return ``None`` to keep content unchanged, a ``str`` to replace content,
    ///     or a dict with a ``"content"`` key.
    ///   name (str): Human-readable name for logging (default "hook").
    ///   error_strategy (str): "skip" (default) or "abort".
    #[pyo3(signature = (stage, handler, name="hook", error_strategy="skip"))]
    fn register_hook(
        &self,
        py: Python<'_>,
        stage: &str,
        handler: Py<PyAny>,
        name: &str,
        error_strategy: &str,
    ) -> PyResult<()> {
        let hook_stage = HookStage::from_str(stage).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid stage {stage:?}; expected one of: pre_completion, post_completion, \
                 pre_tool, post_tool, on_retry, pre_batch_item, post_batch_item"
            ))
        })?;
        let strategy = HookErrorStrategy::from_str(error_strategy).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid error_strategy {error_strategy:?}; expected 'skip' or 'abort'"
            ))
        })?;

        let python_hook = PythonHook {
            callback: Arc::new(handler.clone_ref(py)),
            name: name.to_string(),
        };
        let config = HookConfig {
            name: name.to_string(),
            error_strategy: strategy,
            handler: Arc::new(python_hook),
        };

        let hooks = Arc::clone(&self.hooks);
        runtime().block_on(hooks.add(hook_stage, config));
        Ok(())
    }

    /// Opaque pointer to the internal tool registry (for extension modules).
    fn tools_registry_ptr(&self) -> u64 {
        Arc::as_ptr(&self.registry) as u64
    }

    /// Opaque pointer to the internal hook registry (for extension modules).
    fn hooks_registry_ptr(&self) -> u64 {
        Arc::as_ptr(&self.hooks) as u64
    }

    // ------------------------------------------------------------------
    // Guardrail registration
    // ------------------------------------------------------------------

    /// Register a custom Python guardrail callable.
    ///
    /// Args:
    ///   handler (callable): ``fn(content: str) -> str | None``
    ///     - Return ``None`` or the original string → allow unchanged.
    ///     - Return a new string → allow with transformation (e.g. redaction).
    ///     - Raise any exception → block (the exception message is the reason).
    ///   stage (str): ``"input"``, ``"output"``, or ``"both"`` (default).
    ///   name (str): Human-readable name used in error messages (default "guardrail").
    #[pyo3(signature = (handler, stage="both", name="guardrail"))]
    fn register_guardrail(
        &self,
        py: Python<'_>,
        handler: Py<PyAny>,
        stage: &str,
        name: &str,
    ) -> PyResult<()> {
        let stages: Vec<GuardrailStage> = match stage {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        let cb = Arc::new(handler.clone_ref(py));
        let guardrails = Arc::clone(&self.guardrails);
        let name_s = name.to_string();
        runtime().block_on(async {
            for s in &stages {
                let filter = match s {
                    GuardrailStage::Input => PyGuardrailStageFilter::Input,
                    GuardrailStage::Output => PyGuardrailStageFilter::Output,
                };
                let g = Arc::new(PythonGuardrail {
                    callback: Arc::clone(&cb),
                    stage_filter: filter,
                    name: name_s.clone(),
                }) as Arc<dyn GuardrailHandler>;
                let config = GuardrailConfig {
                    name: name_s.clone(),
                    handler: g,
                };
                match s {
                    GuardrailStage::Input => guardrails.add_input(config).await,
                    GuardrailStage::Output => guardrails.add_output(config).await,
                }
            }
        });
        Ok(())
    }

    /// Register a built-in blocklist guardrail.
    ///
    /// Args:
    ///   patterns (list[str]): List of regex patterns to match.
    ///   action (str): ``"block"`` (default) or ``"redact"``.
    ///   stage (str): ``"input"``, ``"output"``, or ``"both"`` (default).
    ///   name (str): Human-readable name (default "blocklist").
    #[pyo3(signature = (patterns, action="block", stage="both", name="blocklist"))]
    fn add_blocklist_guardrail(
        &self,
        patterns: Vec<String>,
        action: &str,
        stage: &str,
        name: &str,
    ) -> PyResult<()> {
        let bl_action = match action {
            "block" => BlocklistAction::Block,
            "redact" => BlocklistAction::Redact,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid action {other:?}; expected 'block' or 'redact'"
                )));
            }
        };
        let patterns_ref: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
        let mut handler = BlocklistGuardrail::new(&patterns_ref, bl_action)
            .map_err(|e| PyValueError::new_err(format!("invalid regex: {e}")))?;

        let stages = match stage {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        handler = handler.for_stages(stages.clone());

        let config = GuardrailConfig {
            name: name.to_string(),
            handler: Arc::new(handler),
        };
        let guardrails = Arc::clone(&self.guardrails);
        runtime().block_on(async {
            if stages.contains(&GuardrailStage::Input) {
                guardrails
                    .add_input(GuardrailConfig {
                        name: name.to_string(),
                        handler: Arc::new(
                            BlocklistGuardrail::new(&patterns_ref, bl_action.clone())
                                .unwrap()
                                .for_stages(vec![GuardrailStage::Input]),
                        ),
                    })
                    .await;
            }
            if stages.contains(&GuardrailStage::Output) {
                guardrails
                    .add_output(GuardrailConfig {
                        name: name.to_string(),
                        handler: config.handler,
                    })
                    .await;
            }
        });
        Ok(())
    }

    /// Register a built-in max-length guardrail.
    ///
    /// Args:
    ///   max_input (int | None): Maximum characters allowed in the user message.
    ///   max_output (int | None): Maximum characters allowed in the LLM response.
    ///   strategy (str): ``"block"`` (default) or ``"truncate"``.
    ///   name (str): Human-readable name (default "max_length").
    #[pyo3(signature = (max_input=None, max_output=None, strategy="block", name="max_length"))]
    fn add_max_length_guardrail(
        &self,
        max_input: Option<usize>,
        max_output: Option<usize>,
        strategy: &str,
        name: &str,
    ) -> PyResult<()> {
        let len_strategy = match strategy {
            "block" => LengthStrategy::Block,
            "truncate" => LengthStrategy::Truncate,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid strategy {other:?}; expected 'block' or 'truncate'"
                )));
            }
        };
        let guardrails = Arc::clone(&self.guardrails);
        let handler: Arc<dyn GuardrailHandler> =
            Arc::new(MaxLengthGuardrail::new(max_input, max_output, len_strategy));
        let name_s = name.to_string();
        runtime().block_on(async {
            if max_input.is_some() {
                guardrails
                    .add_input(GuardrailConfig {
                        name: name_s.clone(),
                        handler: Arc::clone(&handler),
                    })
                    .await;
            }
            if max_output.is_some() {
                guardrails
                    .add_output(GuardrailConfig {
                        name: name_s,
                        handler: Arc::clone(&handler),
                    })
                    .await;
            }
        });
        Ok(())
    }

    /// Register the built-in PII redaction guardrail (emails, phones, SSNs, credit cards).
    ///
    /// Args:
    ///   stage (str): ``"input"``, ``"output"``, or ``"both"`` (default).
    ///   name (str): Human-readable name (default "pii_redact").
    #[pyo3(signature = (stage="both", name="pii_redact"))]
    fn add_pii_guardrail(&self, stage: &str, name: &str) -> PyResult<()> {
        let stages: Vec<GuardrailStage> = match stage {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        let guardrails = Arc::clone(&self.guardrails);
        let name_s = name.to_string();
        runtime().block_on(async {
            if stages.contains(&GuardrailStage::Input) {
                guardrails
                    .add_input(GuardrailConfig {
                        name: name_s.clone(),
                        handler: Arc::new(
                            PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Input]),
                        ),
                    })
                    .await;
            }
            if stages.contains(&GuardrailStage::Output) {
                guardrails
                    .add_output(GuardrailConfig {
                        name: name_s,
                        handler: Arc::new(
                            PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Output]),
                        ),
                    })
                    .await;
            }
        });
        Ok(())
    }

    // ------------------------------------------------------------------
    // Agent convenience method
    // ------------------------------------------------------------------

    /// Run an agent spec against a user message, sharing this client's HTTP
    /// connection pool, hooks, and guardrails.
    ///
    /// Args:
    ///   spec (AgentSpec): The agent to run.
    ///   user_message (str): The user's message.
    ///   timeout_secs (int | None): Per-call timeout override.
    ///   connect_timeout_secs (int | None): Per-call connect timeout override.
    ///
    /// Returns:
    ///   CompletionOutcome
    #[pyo3(signature = (spec, user_message, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn run_agent(
        &self,
        py: Python<'_>,
        spec: &PyAgentSpec,
        user_message: String,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyCompletionOutcome> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);

        let engine = AgentEngine::new(spec.inner.clone())
            .with_hooks(Arc::clone(&hooks))
            .with_guardrails(Arc::clone(&guardrails));

        // Build options from the client's base options, letting the spec override the model.
        let mut opts = self.options.clone();
        if !spec.inner.model.is_empty() {
            opts.model = spec.inner.model.clone();
        }
        opts.max_tool_rounds = spec.inner.max_tool_rounds;
        opts.request_id = request_id;

        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(engine.run(&http, &registry, user_message, &opts))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = runtime()
            .block_on(engine.run(&http, &registry, user_message, &opts))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        completion_outcome_to_py(py, outcome)
    }

    // ------------------------------------------------------------------

    /// Run a single-turn completion with the registered tools.
    ///
    /// Args:
    ///   user_message (str): The user's message.
    ///   timeout_secs (int | None): Override total request timeout for this call only.
    ///   connect_timeout_secs (int | None): Override connect timeout for this call only.
    ///
    /// Returns:
    ///   CompletionOutcome
    #[pyo3(signature = (user_message, timeout_secs=None, connect_timeout_secs=None, request_id=None, reasoning_effort=None))]
    fn complete(
        &self,
        py: Python<'_>,
        user_message: String,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> PyResult<PyCompletionOutcome> {
        let messages = vec![ChatMessage::text("user", user_message)];

        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let options = finalize_call_options(
            &self.options,
            &self.status_emitter,
            request_id,
            reasoning_effort,
        );

        // With the GIL (classic Python): release it before blocking so that tool
        // callbacks running in spawn_blocking can re-acquire it via Python::attach.
        // With free-threaded Python (Py_GIL_DISABLED): no GIL exists, call directly.
        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            // SAFETY: SaveThread/RestoreThread are called symmetrically. No Python
            // objects are accessed while the GIL is released; tool callbacks
            // re-acquire it themselves inside spawn_blocking via Python::attach.
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(complete_with_tools(
                        &http,
                        &registry,
                        &hooks,
                        &guardrails,
                        messages,
                        &options,
                    ))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = runtime()
            .block_on(complete_with_tools(
                &http,
                &registry,
                &hooks,
                &guardrails,
                messages,
                &options,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        completion_outcome_to_py(py, outcome)
    }

    /// Run a completion with an explicit OpenAI-style ``messages`` list.
    ///
    /// Args:
    ///   messages (list[dict]): Roles ``user``, ``assistant``, ``tool``, etc.
    ///   timeout_secs (int | None): Override total request timeout for this call only.
    ///   connect_timeout_secs (int | None): Override connect timeout for this call only.
    ///   request_id (str | None): Optional correlation ID.
    ///
    /// Returns:
    ///   CompletionOutcome — ``messages`` attribute holds caller-visible history after the turn.
    #[pyo3(signature = (messages, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn complete_messages(
        &self,
        py: Python<'_>,
        messages: Bound<'_, PyAny>,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyCompletionOutcome> {
        let rust_messages = py_to_chat_messages(&messages)?;
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let options = finalize_call_options(
            &self.options,
            &self.status_emitter,
            request_id,
            None,
        );

        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(complete_with_tools(
                        &http,
                        &registry,
                        &hooks,
                        &guardrails,
                        rust_messages,
                        &options,
                    ))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = runtime()
            .block_on(complete_with_tools(
                &http,
                &registry,
                &hooks,
                &guardrails,
                rust_messages,
                &options,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        completion_outcome_to_py(py, outcome)
    }

    /// Stream a chat completion, calling `on_token` for each content delta.
    ///
    /// Unlike `complete()`, this method does **not** execute tool calls. It is intended
    /// for live token-by-token display.
    ///
    /// Args:
    ///   user_message (str): The user's message.
    ///   on_token (callable): Called with each text delta string as it arrives.
    ///   timeout_secs (int | None): Override total request timeout for this call only.
    ///   connect_timeout_secs (int | None): Override connect timeout for this call only.
    ///
    /// Returns:
    ///   StreamOutcome — full accumulated text, finish_reason, and optional usage.
    #[pyo3(signature = (user_message, on_token, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn stream(
        &self,
        py: Python<'_>,
        user_message: String,
        on_token: Py<PyAny>,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyStreamOutcome> {
        eprintln!(
            "[stream] called — Py_GIL_DISABLED={}",
            cfg!(Py_GIL_DISABLED)
        );

        let messages = vec![ChatMessage::text("user", user_message)];
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let mut options = self.options.clone();
        options.request_id = request_id;
        let callback = Arc::new(on_token);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let registry = Arc::clone(&self.registry);
        let has_tools = !runtime().block_on(registry.list_specs()).is_empty();

        // Deltas arrive from an async Tokio task but Python::attach must be
        // called from a real OS thread (not Tokio's async polling context) to
        // avoid a segfault. We route deltas through a channel to a
        // spawn_blocking thread — exactly how tool callbacks work.
        let run_stream = move || {
            let cb = Arc::clone(&callback);
            async move {
                eprintln!("[stream binding] async task started");
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

                // Consumer: receives deltas on a blocking thread and calls Python.
                let consumer = tokio::task::spawn_blocking(move || {
                    eprintln!("[stream binding] consumer thread started");
                    let mut n = 0usize;
                    while let Some(delta) = rx.blocking_recv() {
                        n += 1;
                        eprintln!(
                            "[stream binding] consumer calling Python callback #{n}: {:?}",
                            delta
                        );
                        Python::attach(|py| {
                            let _ = cb.call1(py, (delta,));
                        });
                    }
                    eprintln!("[stream binding] consumer thread done ({n} callbacks)");
                });

                // Producer: runs the SSE stream and sends each delta to the channel.
                // When stream_complete returns, `on_delta` (and tx) are dropped,
                // which closes the channel and lets the consumer finish.
                let on_delta = move |delta: String| {
                    let _ = tx.send(delta);
                };
                let stream_result = if has_tools {
                    stream_complete_with_tools(
                        &http,
                        &registry,
                        &hooks,
                        &guardrails,
                        messages,
                        &options,
                        on_delta,
                    )
                    .await
                    .map(|o| StreamOutcome {
                        content: o.content,
                        finish_reason: o.finish_reason,
                        usage: o.usage,
                        request_id: o.request_id,
                    })
                } else {
                    stream_complete(&http, &hooks, &guardrails, messages, &options, on_delta).await
                };

                // Wait for all queued callbacks to finish before returning.
                let _ = consumer.await;
                stream_result
            }
        };

        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            eprintln!("[stream] taking GIL branch (SaveThread/RestoreThread)");
            // SAFETY: GIL is released before blocking; Python::attach inside
            // spawn_blocking re-acquires it safely on a dedicated OS thread.
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                eprintln!("[stream] GIL saved (tstate={:p})", tstate);
                let result = runtime().block_on(run_stream()).map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = {
            eprintln!("[stream] taking GIL-free branch (Py_GIL_DISABLED)");
            runtime()
                .block_on(run_stream())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };

        Ok(stream_outcome_to_py(py, outcome))
    }

    /// Run a completion via the OpenAI Responses API (`/v1/responses`).
    #[pyo3(signature = (user_message, timeout_secs=None, connect_timeout_secs=None, request_id=None, reasoning_effort=None))]
    fn complete_response(
        &self,
        py: Python<'_>,
        user_message: String,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> PyResult<PyResponseOutcome> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let options = finalize_call_options(
            &self.options,
            &self.status_emitter,
            request_id,
            reasoning_effort,
        );

        let outcome = runtime()
            .block_on(complete_response_with_tools(
                &http,
                &registry,
                &hooks,
                &guardrails,
                user_message,
                &options,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(response_outcome_to_py(py, outcome))
    }

    /// Stream a Responses API completion; `on_token` receives text deltas.
    #[pyo3(signature = (user_message, on_token, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn stream_response(
        &self,
        py: Python<'_>,
        user_message: String,
        on_token: Py<PyAny>,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyResponseStreamOutcome> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let mut options = self.options.clone();
        options.request_id = request_id;
        options.status_emitter = self.status_emitter.clone();
        let callback = Arc::new(on_token);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);

        let run_stream = move || {
            let cb = Arc::clone(&callback);
            async move {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                let consumer = tokio::task::spawn_blocking(move || {
                    while let Some(delta) = rx.blocking_recv() {
                        Python::attach(|py| {
                            let _ = cb.call1(py, (delta,));
                        });
                    }
                });
                let on_delta = move |delta: String| {
                    let _ = tx.send(delta);
                };
                let stream_result =
                    stream_response_api(&http, &hooks, &guardrails, user_message, &options, on_delta)
                        .await;
                let _ = consumer.await;
                stream_result
            }
        };

        let outcome = runtime()
            .block_on(run_stream())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(response_stream_outcome_to_py(py, outcome))
    }

    /// Connect an MCP server over stdio and register its tools.
    #[pyo3(signature = (command, args=None, env=None, prefix=None))]
    fn connect_mcp_stdio(
        &self,
        command: String,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        prefix: Option<String>,
    ) -> PyResult<()> {
        let registry = Arc::clone(&self.registry);
        let session = runtime()
            .block_on(McpSession::connect_stdio(McpStdioConfig {
                command,
                args: args.unwrap_or_default(),
                env,
                label: prefix.clone(),
            }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        runtime()
            .block_on(session.register_tools(&registry, prefix.as_deref()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.mcp_sessions.lock().unwrap().push(session);
        Ok(())
    }

    /// Connect an MCP server over streamable HTTP and register its tools.
    #[pyo3(signature = (url, prefix=None))]
    fn connect_mcp_http(&self, url: String, prefix: Option<String>) -> PyResult<()> {
        let registry = Arc::clone(&self.registry);
        let session = runtime()
            .block_on(McpSession::connect_http(McpHttpConfig {
                url,
                label: prefix.clone(),
            }))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        runtime()
            .block_on(session.register_tools(&registry, prefix.as_deref()))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.mcp_sessions.lock().unwrap().push(session);
        Ok(())
    }

    /// Run multiple prompts concurrently as a batch.
    ///
    /// Args:
    ///   requests (list[PyBatchRequest]): List of batch requests to process.
    ///   max_concurrent (int): Maximum simultaneous in-flight requests (default 5).
    ///   error_strategy (str): One of "continue", "skip", "fail_fast" (default "continue").
    ///
    /// Returns:
    ///   PyBatchResponse
    #[pyo3(signature = (requests, max_concurrent=5, error_strategy="continue",
                        timeout_secs=None, connect_timeout_secs=None))]
    fn batch(
        &self,
        py: Python<'_>,
        requests: Vec<PyRef<'_, PyBatchRequest>>,
        max_concurrent: usize,
        error_strategy: &str,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
    ) -> PyResult<PyBatchResponse> {
        let strategy = ErrorStrategy::from_str(error_strategy)
            .ok_or_else(|| PyValueError::new_err(format!(
                "invalid error_strategy {error_strategy:?}; expected 'continue', 'skip', or 'fail_fast'"
            )))?;

        let batch_requests: Vec<BatchRequest> = requests
            .iter()
            .map(|r| {
                let mut br = BatchRequest::new(r.prompt.clone());
                br.id = r.id.clone();
                br.system_prompt = r.system_prompt.clone();
                br
            })
            .collect();

        let config = BatchConfig {
            max_concurrent,
            error_strategy: strategy,
            timeout: timeout_secs.map(Duration::from_secs),
            connect_timeout: connect_timeout_secs.map(Duration::from_secs),
            cancel: None,
        };

        let http = Arc::clone(&self.http);
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let options = self.options.clone();

        #[cfg(not(Py_GIL_DISABLED))]
        let response = {
            // SAFETY: symmetric SaveThread/RestoreThread; no Python objects accessed while released.
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(batch_complete(
                        http,
                        registry,
                        hooks,
                        guardrails,
                        batch_requests,
                        &options,
                        config,
                    ))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let response = runtime()
            .block_on(batch_complete(
                http,
                registry,
                hooks,
                guardrails,
                batch_requests,
                &options,
                config,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Convert results (GIL is held here).
        let results: Vec<Py<PyBatchResult>> = response
            .results
            .into_iter()
            .map(|r| {
                let usage_py = r.usage.map(|u| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("prompt_tokens", u.prompt_tokens).ok();
                    dict.set_item("completion_tokens", u.completion_tokens).ok();
                    dict.set_item("total_tokens", u.total_tokens).ok();
                    dict.unbind().into_any()
                });
                Py::new(
                    py,
                    PyBatchResult {
                        id: r.id,
                        success: r.success,
                        content: r.content,
                        error: r.error,
                        rounds: r.rounds,
                        usage: usage_py,
                        elapsed_secs: r.elapsed_secs,
                    },
                )
                .unwrap()
            })
            .collect();

        let total_usage_py = response.total_usage.map(|u| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("prompt_tokens", u.prompt_tokens).ok();
            dict.set_item("completion_tokens", u.completion_tokens).ok();
            dict.set_item("total_tokens", u.total_tokens).ok();
            dict.unbind().into_any()
        });

        Ok(PyBatchResponse {
            results,
            total_requests: response.total_requests,
            successful: response.successful,
            failed: response.failed,
            elapsed_secs: response.elapsed_secs,
            total_usage: total_usage_py,
        })
    }
}

// ---------------------------------------------------------------------------
// Conversation (multi-turn)
// ---------------------------------------------------------------------------

/// Stateful chat session sharing a :class:`Client`'s tools, hooks, and HTTP pool.
///
/// Args:
///   client (Client): Source for model settings, tools, hooks, and guardrails.
#[pyclass(name = "Conversation")]
struct PyConversation {
    options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
    turns: Mutex<Vec<ChatMessage>>,
}

#[pymethods]
impl PyConversation {
    #[new]
    fn new(client: &Bound<'_, PyClient>) -> PyResult<Self> {
        let c = client.borrow();
        Ok(Self {
            options: c.options.clone(),
            registry: Arc::clone(&c.registry),
            hooks: Arc::clone(&c.hooks),
            guardrails: Arc::clone(&c.guardrails),
            http: Arc::clone(&c.http),
            turns: Mutex::new(Vec::new()),
        })
    }

    fn push_user(&self, text: String) -> PyResult<()> {
        let mut g = self
            .turns
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        g.push(ChatMessage::text("user", text));
        Ok(())
    }

    /// Append a plain assistant text message (e.g. hand-written protocol text).
    fn push_assistant_text(&self, text: String) -> PyResult<()> {
        let mut g = self
            .turns
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        g.push(ChatMessage::text("assistant", text));
        Ok(())
    }

    /// Caller-visible messages as a list of dicts (JSON-compatible).
    #[getter]
    fn messages(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let g = self
                .turns
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            chat_messages_to_py_any(py, &g)
        })
    }

    /// Run the tool loop for the buffered messages and replace the buffer with
    /// ``outcome.messages`` from the completion result.
    #[pyo3(signature = (timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn complete(
        &self,
        py: Python<'_>,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyCompletionOutcome> {
        let msgs = self
            .turns
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .clone();
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let mut options = self.options.clone();
        options.request_id = request_id;

        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(complete_with_tools(
                        &http,
                        &registry,
                        &hooks,
                        &guardrails,
                        msgs,
                        &options,
                    ))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = runtime()
            .block_on(complete_with_tools(
                &http,
                &registry,
                &hooks,
                &guardrails,
                msgs,
                &options,
            ))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        {
            let mut g = self
                .turns
                .lock()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            *g = outcome.messages.clone();
        }
        completion_outcome_to_py(py, outcome)
    }

    fn __repr__(&self) -> String {
        let n = self.turns.lock().map(|g| g.len()).unwrap_or(0);
        format!("Conversation(messages={n})")
    }
}

// ---------------------------------------------------------------------------
// Batch types
// ---------------------------------------------------------------------------

/// A single request in a batch.
///
/// Attributes:
///   prompt (str): The user message.
///   id (str | None): Optional caller-supplied identifier.
///   system_prompt (str | None): Per-request system prompt override.
#[pyclass(name = "BatchRequest")]
struct PyBatchRequest {
    #[pyo3(get, set)]
    prompt: String,
    #[pyo3(get, set)]
    id: Option<String>,
    #[pyo3(get, set)]
    system_prompt: Option<String>,
}

#[pymethods]
impl PyBatchRequest {
    #[new]
    #[pyo3(signature = (prompt, id=None, system_prompt=None))]
    fn new(prompt: String, id: Option<String>, system_prompt: Option<String>) -> Self {
        PyBatchRequest {
            prompt,
            id,
            system_prompt,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchRequest(prompt={:?}, id={:?}, system_prompt={:?})",
            self.prompt, self.id, self.system_prompt
        )
    }
}

/// Outcome of a single item in a batch.
///
/// Attributes:
///   id (str): Request identifier.
///   success (bool): Whether the completion succeeded.
///   content (str | None): Assistant response text.
///   error (str | None): Error message on failure.
///   rounds (int): Number of completion HTTP calls.
///   usage (dict | None): Token usage.
///   elapsed_secs (float): Wall-clock seconds for this item.
#[pyclass(name = "BatchResult")]
struct PyBatchResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    content: Option<String>,
    #[pyo3(get)]
    error: Option<String>,
    #[pyo3(get)]
    rounds: u32,
    #[pyo3(get)]
    usage: Option<Py<PyAny>>,
    #[pyo3(get)]
    elapsed_secs: f64,
}

#[pymethods]
impl PyBatchResult {
    fn __repr__(&self) -> String {
        format!(
            "BatchResult(id={:?}, success={}, content={:?}, error={:?})",
            self.id, self.success, self.content, self.error
        )
    }
}

/// Aggregate outcome of a batch call.
///
/// Attributes:
///   results (list[BatchResult]): Per-item results.
///   total_requests (int): Total number of input requests.
///   successful (int): Number that succeeded.
///   failed (int): Number that failed.
///   elapsed_secs (float): Total wall-clock seconds.
///   total_usage (dict | None): Aggregated token usage.
#[pyclass(name = "BatchResponse")]
struct PyBatchResponse {
    #[pyo3(get)]
    results: Vec<Py<PyBatchResult>>,
    #[pyo3(get)]
    total_requests: usize,
    #[pyo3(get)]
    successful: usize,
    #[pyo3(get)]
    failed: usize,
    #[pyo3(get)]
    elapsed_secs: f64,
    #[pyo3(get)]
    total_usage: Option<Py<PyAny>>,
}

#[pymethods]
impl PyBatchResponse {
    fn __repr__(&self) -> String {
        format!(
            "BatchResponse(total={}, successful={}, failed={}, elapsed={:.2}s)",
            self.total_requests, self.successful, self.failed, self.elapsed_secs
        )
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
// AgentSpec Python binding
// ---------------------------------------------------------------------------

/// Declarative configuration for a superglue agent.
///
/// Args:
///   name (str): Unique identifier for this agent (used in logs).
///   persona (str): Personality / role description.
///   goals (list[str]): What the agent aims to achieve.
///   constraints (list[str]): Rules the agent must not violate.
///   model (str): LLM model override (e.g. "gpt-5.4-nano-2026-03-17"). Empty string keeps
///     the default from the Client.
///   max_tool_rounds (int): Max tool-call iterations per run (default 16).
///   max_output_retries (int): Max output-guardrail retries (default 3).
///   system_prompt (str | None): Verbatim system prompt — skips compilation
///     of persona/goals/constraints when set.
#[pyclass(name = "AgentSpec")]
#[derive(Clone)]
struct PyAgentSpec {
    inner: AgentSpec,
}

#[pymethods]
impl PyAgentSpec {
    #[new]
    #[pyo3(signature = (
        name,
        persona,
        goals = None,
        constraints = None,
        model = "",
        max_tool_rounds = 16u32,
        max_output_retries = 3u32,
        system_prompt = None,
        reasoning_effort = None,
    ))]
    fn new(
        name: String,
        persona: String,
        goals: Option<Vec<String>>,
        constraints: Option<Vec<String>>,
        model: &str,
        max_tool_rounds: u32,
        max_output_retries: u32,
        system_prompt: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Self {
        let mut spec = AgentSpec::new(name, persona)
            .with_model(model)
            .with_max_tool_rounds(max_tool_rounds);
        spec.max_output_retries = max_output_retries;
        spec.reasoning_effort = reasoning_effort;
        if let Some(g) = goals {
            spec.goals = g;
        }
        if let Some(c) = constraints {
            spec.constraints = c;
        }
        if let Some(sp) = system_prompt {
            spec = spec.with_system_prompt(sp);
        }
        PyAgentSpec { inner: spec }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn persona(&self) -> &str {
        &self.inner.persona
    }
    #[getter]
    fn goals(&self) -> Vec<String> {
        self.inner.goals.clone()
    }
    #[getter]
    fn constraints(&self) -> Vec<String> {
        self.inner.constraints.clone()
    }
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn max_tool_rounds(&self) -> u32 {
        self.inner.max_tool_rounds
    }
    #[getter]
    fn max_output_retries(&self) -> u32 {
        self.inner.max_output_retries
    }
    #[getter]
    fn reasoning_effort(&self) -> Option<String> {
        self.inner.reasoning_effort.clone()
    }

    fn compile_system_prompt(&self) -> String {
        self.inner.compile_system_prompt()
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentSpec(name={:?}, persona={:?}, goals={:?}, constraints={:?})",
            self.inner.name, self.inner.persona, self.inner.goals, self.inner.constraints
        )
    }
}

// ---------------------------------------------------------------------------
// AgentEngine Python binding
// ---------------------------------------------------------------------------

/// Execution engine for a superglue agent.
///
/// Compiles the agent's persona/goals/constraints into a system prompt and
/// runs `complete_with_tools` (or `stream_complete`) with that prompt
/// pre-injected. Hooks and guardrails registered on this engine are layered
/// on top of the underlying pipeline.
///
/// Args:
///   spec (AgentSpec): The agent's configuration.
///   api_key (str): OpenAI-compatible API key.
///   base_url (str): API base URL (default: https://api.openai.com).
///   model (str): Model to use when spec.model is empty (default: gpt-5.4-nano-2026-03-17-mini).
///   max_retries (int): HTTP retry attempts (default 3).
///   requests_per_second (int | None): QPS cap (default unlimited).
///   timeout_secs (int): Total request timeout seconds (default 60).
///   connect_timeout_secs (int): TCP connect timeout seconds (default 30).
#[pyclass(name = "AgentEngine")]
struct PyAgentEngine {
    spec: AgentSpec,
    base_options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
    status_emitter: Option<Arc<StatusEmitter>>,
}

#[pymethods]
impl PyAgentEngine {
    #[new]
    #[pyo3(signature = (
        spec,
        api_key,
        base_url = "https://api.openai.com",
        model = "gpt-5.4-nano-2026-03-17-mini",
        max_retries = 3u32,
        retry_initial_delay_ms = 50u64,
        retry_max_delay_ms = 2000u64,
        retry_multiplier = 2.0f64,
        requests_per_second = None,
        timeout_secs = 60u64,
        connect_timeout_secs = 30u64,
        pool_max_idle_per_host = 50usize,
        pool_idle_timeout_secs = None,
        reasoning_effort = None,
        status_emitter = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: &PyAgentSpec,
        api_key: String,
        base_url: &str,
        model: &str,
        max_retries: u32,
        retry_initial_delay_ms: u64,
        retry_max_delay_ms: u64,
        retry_multiplier: f64,
        requests_per_second: Option<u32>,
        timeout_secs: u64,
        connect_timeout_secs: u64,
        pool_max_idle_per_host: usize,
        pool_idle_timeout_secs: Option<u64>,
        reasoning_effort: Option<String>,
        status_emitter: Option<PyStatusEmitter>,
    ) -> PyResult<Self> {
        let cfg = ClientConfig {
            retry: RetryPolicy {
                max_retries,
                initial_interval_ms: retry_initial_delay_ms,
                max_interval_ms: retry_max_delay_ms,
                multiplier: retry_multiplier,
            },
            quota_per_second: requests_per_second.and_then(NonZeroU32::new),
            timeout: Duration::from_secs(timeout_secs),
            connect_timeout: Duration::from_secs(connect_timeout_secs),
            pool_max_idle_per_host,
            pool_idle_timeout: pool_idle_timeout_secs.map(Duration::from_secs),
            ..ClientConfig::default()
        };
        let http = HttpClient::new(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let effective_model = if spec.inner.model.is_empty() {
            model.to_string()
        } else {
            spec.inner.model.clone()
        };

        let guardrails = Arc::new(
            GuardrailRegistry::new().with_max_output_retries(spec.inner.max_output_retries),
        );

        Ok(PyAgentEngine {
            spec: spec.inner.clone(),
            base_options: ChatOptions {
                api_key: Secret::new(api_key),
                model: effective_model,
                base_url: base_url.to_string(),
                max_tool_rounds: spec.inner.max_tool_rounds,
                reasoning_effort: reasoning_effort.or(spec.inner.reasoning_effort.clone()),
                ..Default::default()
            },
            registry: Arc::new(ToolRegistry::new()),
            hooks: Arc::new(HookRegistry::new()),
            guardrails,
            http: Arc::new(http),
            status_emitter: status_emitter.map(|e| e.inner),
        })
    }

    // --- Tool registration (same API as PyClient) ---

    #[pyo3(signature = (name, description, parameters, r#fn))]
    fn register_tool(
        &self,
        py: Python<'_>,
        name: String,
        description: String,
        parameters: Bound<'_, PyAny>,
        r#fn: Py<PyAny>,
    ) -> PyResult<()> {
        let json_mod = py.import("json")?;
        let params_json: String = json_mod.call_method1("dumps", (&parameters,))?.extract()?;
        let parameters_schema: Value =
            serde_json::from_str(&params_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let spec = ToolSpec {
            name,
            description: Some(description),
            parameters_schema,
        };
        let callback = Arc::new(r#fn.clone_ref(py));
        let tool = Arc::new(PythonDictTool { spec, callback }) as Arc<dyn Tool>;
        runtime()
            .block_on(self.registry.register(tool))
            .map_err(tool_error_to_py)
    }

    /// Register a lifecycle hook on this agent engine.
    #[pyo3(signature = (stage, handler, name="hook", error_strategy="skip"))]
    fn register_hook(
        &self,
        py: Python<'_>,
        stage: &str,
        handler: Py<PyAny>,
        name: &str,
        error_strategy: &str,
    ) -> PyResult<()> {
        let hook_stage = HookStage::from_str(stage).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid stage {stage:?}; expected one of: pre_completion, post_completion, \
                 pre_tool, post_tool, on_retry, pre_batch_item, post_batch_item"
            ))
        })?;
        let strategy = HookErrorStrategy::from_str(error_strategy).ok_or_else(|| {
            PyValueError::new_err(format!(
                "invalid error_strategy {error_strategy:?}; expected 'skip' or 'abort'"
            ))
        })?;
        let python_hook = PythonHook {
            callback: Arc::new(handler.clone_ref(py)),
            name: name.to_string(),
        };
        let config = HookConfig {
            name: name.to_string(),
            error_strategy: strategy,
            handler: Arc::new(python_hook),
        };
        let hooks = Arc::clone(&self.hooks);
        runtime().block_on(hooks.add(hook_stage, config));
        Ok(())
    }

    /// Register a guardrail on this agent engine.
    #[pyo3(signature = (handler, stage="both", name="guardrail"))]
    fn register_guardrail(
        &self,
        py: Python<'_>,
        handler: Py<PyAny>,
        stage: &str,
        name: &str,
    ) -> PyResult<()> {
        let stages: Vec<GuardrailStage> = match stage {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        let cb = Arc::new(handler.clone_ref(py));
        let guardrails = Arc::clone(&self.guardrails);
        let name_s = name.to_string();
        runtime().block_on(async {
            for s in &stages {
                let filter = match s {
                    GuardrailStage::Input => PyGuardrailStageFilter::Input,
                    GuardrailStage::Output => PyGuardrailStageFilter::Output,
                };
                let g = Arc::new(PythonGuardrail {
                    callback: Arc::clone(&cb),
                    stage_filter: filter,
                    name: name_s.clone(),
                }) as Arc<dyn GuardrailHandler>;
                let config = GuardrailConfig {
                    name: name_s.clone(),
                    handler: g,
                };
                match s {
                    GuardrailStage::Input => guardrails.add_input(config).await,
                    GuardrailStage::Output => guardrails.add_output(config).await,
                }
            }
        });
        Ok(())
    }

    // --- Execution ---

    /// Run a single-turn completion with the agent's compiled system prompt.
    ///
    /// Args:
    ///   user_message (str): The user's message.
    ///   timeout_secs (int | None): Override request timeout for this call.
    ///   connect_timeout_secs (int | None): Override connect timeout for this call.
    ///
    /// Returns:
    ///   CompletionOutcome
    #[pyo3(signature = (user_message, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn run(
        &self,
        py: Python<'_>,
        user_message: String,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyCompletionOutcome> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);

        let engine = AgentEngine::new(self.spec.clone())
            .with_hooks(hooks)
            .with_guardrails(guardrails);

        let opts = finalize_call_options(
            &self.base_options,
            &self.status_emitter,
            request_id,
            None,
        );

        #[cfg(not(Py_GIL_DISABLED))]
        let outcome = {
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime()
                    .block_on(engine.run(&http, &registry, user_message, &opts))
                    .map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let outcome = runtime()
            .block_on(engine.run(&http, &registry, user_message, &opts))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        completion_outcome_to_py(py, outcome)
    }

    /// Run a streaming completion with the agent's compiled system prompt.
    ///
    /// Args:
    ///   user_message (str): The user's message.
    ///   on_token (callable): Called with each content token as it arrives.
    ///   timeout_secs (int | None): Override request timeout.
    ///   connect_timeout_secs (int | None): Override connect timeout.
    ///
    /// Returns:
    ///   StreamOutcome
    #[pyo3(signature = (user_message, on_token, timeout_secs=None, connect_timeout_secs=None, request_id=None))]
    fn stream(
        &self,
        py: Python<'_>,
        user_message: String,
        on_token: Py<PyAny>,
        timeout_secs: Option<u64>,
        connect_timeout_secs: Option<u64>,
        request_id: Option<String>,
    ) -> PyResult<PyStreamOutcome> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let callback = Arc::new(on_token);
        let mut opts = self.base_options.clone();
        opts.request_id = request_id;
        let spec = self.spec.clone();

        let run_stream = move || {
            let cb = Arc::clone(&callback);
            async move {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
                let consumer = tokio::task::spawn_blocking(move || {
                    while let Some(delta) = rx.blocking_recv() {
                        Python::attach(|py| {
                            let _ = cb.call1(py, (delta,));
                        });
                    }
                });
                let on_delta = move |delta: String| {
                    let _ = tx.send(delta);
                };
                let engine = AgentEngine::new(spec)
                    .with_hooks(hooks)
                    .with_guardrails(guardrails);
                let stream_result = engine.stream(&http, user_message, &opts, on_delta).await;
                let _ = consumer.await;
                stream_result
            }
        };

        #[cfg(not(Py_GIL_DISABLED))]
        let stream_outcome = {
            unsafe {
                let tstate = pyo3::ffi::PyEval_SaveThread();
                let result = runtime().block_on(run_stream()).map_err(|e| e.to_string());
                pyo3::ffi::PyEval_RestoreThread(tstate);
                result
            }
            .map_err(PyRuntimeError::new_err)?
        };

        #[cfg(Py_GIL_DISABLED)]
        let stream_outcome = runtime()
            .block_on(run_stream())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let usage_py = stream_outcome.usage.map(|u| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("prompt_tokens", u.prompt_tokens).ok();
            dict.set_item("completion_tokens", u.completion_tokens).ok();
            dict.set_item("total_tokens", u.total_tokens).ok();
            dict.unbind().into_any()
        });
        Ok(PyStreamOutcome {
            content: stream_outcome.content,
            finish_reason: stream_outcome.finish_reason,
            usage: usage_py,
            request_id: stream_outcome.request_id,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentEngine(name={:?}, model={:?})",
            self.spec.name, self.spec.model
        )
    }
}

// ---------------------------------------------------------------------------

#[pyfunction]
fn version() -> &'static str {
    superglue::version()
}

#[pymodule(gil_used = false)]
fn _superglue(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing(TelemetryConfig::default());

    m.add_class::<PyUploadedFile>()?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyConversation>()?;
    m.add_class::<PyCompletionOutcome>()?;
    m.add_class::<PyStreamOutcome>()?;
    m.add_class::<PyResponseOutcome>()?;
    m.add_class::<PyResponseStreamOutcome>()?;
    m.add_class::<PyBatchRequest>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchResponse>()?;
    m.add_class::<PyAgentSpec>()?;
    m.add_class::<PyAgentEngine>()?;
    m.add_class::<PyStatusEmitter>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
