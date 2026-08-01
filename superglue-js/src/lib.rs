#![deny(clippy::all)]

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use napi::Status;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

type StdResult<T, E> = std::result::Result<T, E>;
use secrecy::SecretString;
use serde_json::{Value, json};
use superglue::agents::{AgentEngine as SgAgentEngine, AgentSpec};
use superglue::batch::{BatchConfig, BatchRequest, ErrorStrategy, batch_complete};
use superglue::chat::{
    ChatOptions, CompletionOutcome, StreamOutcome, complete_with_tools, stream_complete,
    stream_complete_with_tools,
};
use superglue::client::{
    BindingBootstrapConfig, ClientBuildError, bootstrap_from_parts, provider_id_from_str,
};
use superglue::events::{ProcessEvent, StatusEmitter as SgStatusEmitter, StatusSubscriber};
use superglue::fallback::ModelFallbackChain;
use superglue::files::{self, FilePurpose};
use superglue::guardrails::{
    BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailHandler, GuardrailOutcome,
    GuardrailRegistry, GuardrailStage, LengthStrategy, MaxLengthGuardrail, PiiRedactGuardrail,
};
use superglue::hooks::{
    HookConfig, HookContext, HookError, HookErrorStrategy, HookHandler, HookRegistry, HookStage,
};
use superglue::http::{ClientConfig, HttpClient, RetryPolicy};
use superglue::mcp::{McpHttpConfig, McpSession, McpStdioConfig};
use superglue::openai::ChatMessage;
use superglue::providers::ProviderId;
use superglue::responses::{
    complete_with_tools as complete_response_with_tools, stream_response as stream_response_api,
};
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

// ---------------------------------------------------------------------------
// JSON threadsafe callbacks (args object in, result object out; sync or async on JS side)
// ---------------------------------------------------------------------------

/// JS returns `Promise<serde_json::Value>` so napi awaits the Promise. Using `Value` alone
/// would coerce a returned Promise object to JSON without awaiting, and later rejections/throws
/// become unhandled (see napi-rs ThreadsafeFunction + `serde_json::Value::from_napi_value`).
///
/// `Weak = true`: threadsafe functions do not keep the Node event loop running after the script
/// finishes (same issue as std `ThreadsafeFunction` default otherwise — tools/hooks would require Ctrl+C).
type JsonCallback = ThreadsafeFunction<Value, Promise<Value>, Value, Status, true, true, 0>;

/// Process event callback: receives a JSON event object.
type StatusCallback = ThreadsafeFunction<Value, (), Value, Status, true, true, 0>;

fn process_event_to_value(event: &ProcessEvent) -> Value {
    let usage = if let Some(u) = &event.usage {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    } else {
        Value::Null
    };
    json!({
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
}

struct JsStatusSubscriber {
    callback: StatusCallback,
}

#[async_trait]
impl StatusSubscriber for JsStatusSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        let v = process_event_to_value(&event);
        self.callback
            .call(Ok(v), ThreadsafeFunctionCallMode::NonBlocking);
    }
}

#[napi]
#[derive(Clone)]
pub struct StatusEmitterJs {
    inner: Arc<SgStatusEmitter>,
}

#[napi]
impl StatusEmitterJs {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SgStatusEmitter::new()),
        }
    }

    #[napi]
    pub fn subscribe(&self, callback: StatusCallback) {
        let subscriber = Arc::new(JsStatusSubscriber { callback });
        let inner = Arc::clone(&self.inner);
        napi::bindgen_prelude::block_on(async move {
            inner
                .subscribe(subscriber as Arc<dyn StatusSubscriber>)
                .await;
        });
    }
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

/// JS guardrail callbacks `(stageStr, content)` → allow string or structured result.
type GuardrailCallback =
    ThreadsafeFunction<(String, String), Value, (String, String), Status, true, true, 0>;

struct JsDictTool {
    spec: ToolSpec,
    callback: JsonCallback,
}

#[async_trait]
impl Tool for JsDictTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> StdResult<Value, ToolInvokeError> {
        match self.callback.call_async(Ok(arguments)).await {
            Ok(promise) => promise
                .await
                .map_err(|e| ToolInvokeError::handler(e.to_string(), None)),
            Err(e) => Err(ToolInvokeError::handler(e.to_string(), None)),
        }
    }
}

struct JsHook {
    callback: JsonCallback,
    name: String,
}

#[async_trait]
impl HookHandler for JsHook {
    async fn execute(&self, ctx: HookContext) -> StdResult<HookContext, HookError> {
        let stage_str = match ctx.stage {
            HookStage::PreCompletion => "pre_completion",
            HookStage::PostCompletion => "post_completion",
            HookStage::PreTool => "pre_tool",
            HookStage::PostTool => "post_tool",
            HookStage::OnRetry => "on_retry",
            HookStage::PreBatchItem => "pre_batch_item",
            HookStage::PostBatchItem => "post_batch_item",
        };
        let meta = match serde_json::to_value(&ctx.metadata) {
            Ok(v) => v,
            Err(e) => return Err(HookError::new(&self.name, e.to_string())),
        };
        let ctx_obj = json!({
            "stage": stage_str,
            "content": ctx.content,
            "metadata": meta,
        });
        let result = match self.callback.call_async(Ok(ctx_obj)).await {
            Ok(promise) => match promise.await {
                Ok(v) => v,
                Err(e) => return Err(HookError::new(&self.name, e.to_string())),
            },
            Err(e) => return Err(HookError::new(&self.name, e.to_string())),
        };

        let new_content: Option<String> = if result.is_null() {
            None
        } else if let Some(s) = result.as_str() {
            Some(s.to_string())
        } else if let Some(obj) = result.as_object() {
            obj.get("content")
                .and_then(|v| v.as_str())
                .map(String::from)
        } else {
            None
        };

        let content = new_content.unwrap_or(ctx.content);
        Ok(HookContext {
            stage: ctx.stage,
            content,
            metadata: ctx.metadata,
        })
    }
}

enum JsGuardrailStageFilter {
    Input,
    Output,
}

struct JsGuardrail {
    callback: Arc<GuardrailCallback>,
    stage_filter: JsGuardrailStageFilter,
    name: String,
}

#[async_trait]
impl GuardrailHandler for JsGuardrail {
    async fn check(&self, stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let applies = matches!(
            (&self.stage_filter, &stage),
            (JsGuardrailStageFilter::Input, GuardrailStage::Input)
                | (JsGuardrailStageFilter::Output, GuardrailStage::Output)
        );
        if !applies {
            return GuardrailOutcome::Allow(content.to_string());
        }

        let stage_str = match stage {
            GuardrailStage::Input => "input",
            GuardrailStage::Output => "output",
        };
        let content_owned = content.to_string();
        let name = self.name.clone();

        match self
            .callback
            .call_async(Ok((stage_str.to_string(), content_owned.clone())))
            .await
        {
            Ok(v) => {
                if v.is_null() {
                    GuardrailOutcome::Allow(content_owned)
                } else if let Some(s) = v.as_str() {
                    GuardrailOutcome::Allow(s.to_string())
                } else {
                    GuardrailOutcome::Allow(content_owned)
                }
            }
            Err(e) => GuardrailOutcome::Block(format!("{name}: {e}")),
        }
    }
}

fn tool_error_to_napi(e: ToolInvokeError) -> Error {
    Error::from_reason(e.to_string())
}

fn chat_messages_from_value(messages: Value) -> Result<Vec<ChatMessage>> {
    serde_json::from_value(messages).map_err(|e| {
        Error::from_reason(format!(
            "messages must be JSON-serializable OpenAI-style message objects: {e}"
        ))
    })
}

fn completion_outcome_to_js(outcome: CompletionOutcome) -> Result<CompletionOutcomeJs> {
    let usage = outcome.usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });
    let messages =
        serde_json::to_value(&outcome.messages).map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(CompletionOutcomeJs {
        content: outcome.content,
        rounds: outcome.rounds,
        usage,
        request_id: outcome.request_id,
        model_used: outcome.model_used,
        messages,
    })
}

fn effective_http(
    base: &Arc<HttpClient>,
    timeout_secs: Option<i64>,
    connect_timeout_secs: Option<i64>,
) -> Result<Arc<HttpClient>> {
    if timeout_secs.is_none() && connect_timeout_secs.is_none() {
        return Ok(Arc::clone(base));
    }
    let t = timeout_secs
        .map(|s| Duration::from_secs(s.max(0) as u64))
        .unwrap_or(base.config.timeout);
    let ct = connect_timeout_secs
        .map(|s| Duration::from_secs(s.max(0) as u64))
        .unwrap_or(base.config.connect_timeout);
    base.clone_with_timeouts(t, ct)
        .map(Arc::new)
        .map_err(|e| Error::from_reason(e.to_string()))
}

fn stream_outcome_to_js(o: StreamOutcome) -> StreamOutcomeJs {
    let usage = o.usage.map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });
    StreamOutcomeJs {
        content: o.content,
        finish_reason: o.finish_reason,
        usage,
        request_id: o.request_id,
    }
}

// ---------------------------------------------------------------------------
// N-API objects
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct CompletionOutcomeJs {
    pub content: Option<String>,
    pub rounds: u32,
    pub usage: Option<Value>,
    pub request_id: String,
    pub model_used: String,
    pub messages: Value,
}

#[napi(object)]
pub struct ResponseOutcomeJs {
    pub id: String,
    pub content: Option<String>,
    pub rounds: u32,
    pub usage: Option<Value>,
    pub request_id: String,
    pub model_used: String,
}

#[napi(object)]
pub struct ResponseStreamOutcomeJs {
    pub id: String,
    pub content: String,
    pub usage: Option<Value>,
    pub request_id: String,
    pub model_used: String,
}

#[napi(object)]
pub struct StreamOutcomeJs {
    pub content: String,
    pub finish_reason: Option<String>,
    pub usage: Option<Value>,
    pub request_id: String,
}

#[napi(object)]
pub struct BatchRequestJs {
    pub prompt: String,
    pub id: Option<String>,
    pub system_prompt: Option<String>,
}

#[napi(object)]
pub struct BatchResultJs {
    pub id: String,
    pub success: bool,
    pub content: Option<String>,
    pub error: Option<String>,
    pub rounds: u32,
    pub usage: Option<Value>,
    pub elapsed_secs: f64,
}

#[napi(object)]
pub struct BatchResponseJs {
    pub results: Vec<BatchResultJs>,
    pub total_requests: u32,
    pub successful: u32,
    pub failed: u32,
    pub elapsed_secs: f64,
    pub total_usage: Option<Value>,
}

#[napi]
#[derive(Clone)]
pub struct AgentSpecJs {
    pub(crate) inner: AgentSpec,
}

#[napi]
impl AgentSpecJs {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        persona: String,
        goals: Option<Vec<String>>,
        constraints: Option<Vec<String>>,
        #[napi(ts_arg_type = "string")] model: String,
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
        Self { inner: spec }
    }

    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[napi(getter)]
    pub fn persona(&self) -> String {
        self.inner.persona.clone()
    }

    #[napi(getter)]
    pub fn goals(&self) -> Vec<String> {
        self.inner.goals.clone()
    }

    #[napi(getter)]
    pub fn constraints(&self) -> Vec<String> {
        self.inner.constraints.clone()
    }

    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    #[napi(getter)]
    pub fn max_tool_rounds(&self) -> u32 {
        self.inner.max_tool_rounds
    }

    #[napi(getter)]
    pub fn max_output_retries(&self) -> u32 {
        self.inner.max_output_retries
    }

    #[napi(getter)]
    pub fn reasoning_effort(&self) -> Option<String> {
        self.inner.reasoning_effort.clone()
    }

    #[napi]
    pub fn compile_system_prompt(&self) -> String {
        self.inner.compile_system_prompt()
    }
}

struct ClientState {
    options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
    max_upload_bytes: usize,
    mcp_sessions: Mutex<Vec<Arc<McpSession>>>,
}

#[napi(object)]
pub struct UploadedFileJs {
    pub provider: String,
    pub file_id: String,
    pub filename: String,
    pub bytes: u32,
    pub purpose: String,
}

fn file_purpose_from_str(s: &str) -> StdResult<FilePurpose, Error> {
    match s {
        "assistants" => Ok(FilePurpose::Assistants),
        "user_data" => Ok(FilePurpose::UserData),
        "batch" => Ok(FilePurpose::Batch),
        other => Err(Error::from_reason(format!("unknown file purpose: {other}"))),
    }
}

fn provider_credentials_from_map(
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

fn provider_qps_from_map(map: &HashMap<String, u32>) -> HashMap<ProviderId, u32> {
    let mut out = HashMap::new();
    for (name, qps) in map {
        if let Some(pid) = provider_id_from_str(name) {
            out.insert(pid, *qps);
        }
    }
    out
}

#[napi]
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientState>,
}

#[napi]
impl Client {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: String,
        #[napi(ts_arg_type = "string | undefined")] model: Option<String>,
        #[napi(ts_arg_type = "string | undefined")] base_url: Option<String>,
        system_prompt: Option<String>,
        max_tool_rounds: Option<u32>,
        max_retries: Option<u32>,
        retry_initial_delay_ms: Option<u32>,
        retry_max_delay_ms: Option<u32>,
        retry_multiplier: Option<f64>,
        requests_per_second: Option<u32>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        max_output_retries: Option<u32>,
        pool_max_idle_per_host: Option<u32>,
        pool_idle_timeout_secs: Option<i64>,
        reasoning_effort: Option<String>,
        status_emitter: Option<&StatusEmitterJs>,
        model_fallback_models: Option<Vec<String>>,
        api_keys: Option<HashMap<String, String>>,
        requests_per_second_for: Option<HashMap<String, u32>>,
        max_upload_bytes: Option<u32>,
        #[napi(ts_arg_type = "string | undefined")] tool_mode: Option<String>,
        tool_route_model: Option<String>,
        condense_tool_messages: Option<bool>,
        aaak_tool_condensing: Option<bool>,
        summarize_context_enabled: Option<bool>,
        summarize_context_threshold: Option<u32>,
        summarize_context_keep_recent: Option<u32>,
        aaak_compression_enabled: Option<bool>,
        aaak_compression_model: Option<String>,
    ) -> Result<Self> {
        let max_retries = max_retries.unwrap_or(3);
        let retry_initial_delay_ms = retry_initial_delay_ms.unwrap_or(50);
        let retry_max_delay_ms = retry_max_delay_ms.unwrap_or(2000);
        let retry_multiplier = retry_multiplier.unwrap_or(2.0);
        let max_output_retries = max_output_retries.unwrap_or(3);
        let status_emitter_arc = status_emitter.map(|e| Arc::clone(&e.inner));
        let model_fallback = model_fallback_models.and_then(|models| {
            if models.is_empty() {
                None
            } else {
                Some(ModelFallbackChain::new(models))
            }
        });
        let provider_creds = api_keys
            .filter(|m| !m.is_empty())
            .map(|m| provider_credentials_from_map(&m));

        let bootstrap = bootstrap_from_parts(BindingBootstrapConfig {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com".to_string()),
            model: model.unwrap_or_else(|| "gpt-5.4-nano-2026-03-17-mini".to_string()),
            system_prompt,
            max_tool_rounds: max_tool_rounds.unwrap_or(16),
            max_retries,
            retry_initial_delay_ms: u64::from(retry_initial_delay_ms),
            retry_max_delay_ms: u64::from(retry_max_delay_ms),
            retry_multiplier,
            requests_per_second,
            timeout: Duration::from_secs(timeout_secs.unwrap_or(60).max(0) as u64),
            connect_timeout: Duration::from_secs(connect_timeout_secs.unwrap_or(30).max(0) as u64),
            max_output_retries,
            pool_max_idle_per_host: pool_max_idle_per_host.unwrap_or(50) as usize,
            pool_idle_timeout: pool_idle_timeout_secs.map(|s| Duration::from_secs(s.max(0) as u64)),
            reasoning_effort,
            status_emitter: status_emitter_arc,
            model_fallback,
            provider_credentials: provider_creds,
            provider_qps: provider_qps_from_map(&requests_per_second_for.unwrap_or_default()),
            max_upload_bytes: max_upload_bytes
                .map(|b| b as usize)
                .unwrap_or_else(files::default_max_upload_bytes),
        })
        .map_err(|e: ClientBuildError| Error::from_reason(e.to_string()))?;

        let mut options = bootstrap.options;
        options.tool_mode = match tool_mode.as_deref() {
            Some("dynamic") => superglue::tools::ToolMode::Dynamic,
            _ => superglue::tools::ToolMode::Standard,
        };
        options.tool_route_model = tool_route_model;
        options.condense_tool_messages = condense_tool_messages.unwrap_or(false);
        options.aaak_tool_condensing = aaak_tool_condensing.unwrap_or(false);
        options.summarize_context = superglue::context::SummarizeContextConfig {
            enabled: summarize_context_enabled.unwrap_or(false),
            threshold: summarize_context_threshold.unwrap_or(20) as usize,
            keep_recent: summarize_context_keep_recent.unwrap_or(12) as usize,
            max_chars: 800_000,
        };
        options.aaak_compression_enabled = aaak_compression_enabled.unwrap_or(false);
        options.aaak_compression_model = aaak_compression_model;

        Ok(Self {
            inner: Arc::new(ClientState {
                options,
                registry: Arc::new(ToolRegistry::new()),
                hooks: Arc::new(HookRegistry::new()),
                guardrails: Arc::new(
                    GuardrailRegistry::new().with_max_output_retries(max_output_retries),
                ),
                http: Arc::new(bootstrap.http),
                max_upload_bytes: bootstrap.max_upload_bytes,
                mcp_sessions: Mutex::new(Vec::new()),
            }),
        })
    }

    #[napi]
    pub async fn upload_file(
        &self,
        path: String,
        purpose: Option<String>,
        provider: Option<String>,
    ) -> Result<UploadedFileJs> {
        let purpose = file_purpose_from_str(purpose.as_deref().unwrap_or("user_data"))?;
        let provider_id = provider
            .as_deref()
            .and_then(provider_id_from_str)
            .unwrap_or_else(|| {
                superglue::providers::parse_model_ref(&self.inner.options.model).provider
            });
        let creds = superglue::chat::credentials_for(&self.inner.options);
        let uploaded = files::upload_file(
            &self.inner.http,
            &creds,
            provider_id,
            std::path::Path::new(&path),
            purpose,
            self.inner.max_upload_bytes,
        )
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(UploadedFileJs {
            provider: uploaded.provider.to_string(),
            file_id: uploaded.file_id,
            filename: uploaded.filename,
            bytes: uploaded.bytes as u32,
            purpose: uploaded.purpose.as_openai_str().to_string(),
        })
    }

    #[napi]
    pub fn message_with_file_bytes(
        filename: String,
        file_bytes: Buffer,
        text: Option<String>,
    ) -> Result<Value> {
        let msg = files::message_with_file_bytes(text.as_deref(), &filename, file_bytes.as_ref());
        serde_json::to_value(msg).map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Register a tool: `callback` receives JSON args and returns a JSON object (or Promise of).
    #[napi]
    pub async fn register_tool(
        &self,
        name: String,
        description: String,
        parameters: Value,
        callback: JsonCallback,
        static_tool: Option<bool>,
    ) -> Result<()> {
        let spec = ToolSpec {
            name,
            description: Some(description),
            parameters_schema: parameters,
            static_tool: static_tool.unwrap_or(false),
        };
        let tool = Arc::new(JsDictTool { spec, callback }) as Arc<dyn Tool>;
        self.inner
            .registry
            .register(tool)
            .await
            .map_err(tool_error_to_napi)
    }

    #[napi]
    pub async fn register_hook(
        &self,
        stage: String,
        handler: JsonCallback,
        #[napi(ts_arg_type = "string | undefined")] name: Option<String>,
        error_strategy: Option<String>,
    ) -> Result<()> {
        let hook_stage = HookStage::from_str(&stage).ok_or_else(|| {
            Error::from_reason(format!(
                "invalid stage {stage:?}; expected pre_completion, post_completion, pre_tool, post_tool, on_retry, pre_batch_item, post_batch_item"
            ))
        })?;
        let strategy = HookErrorStrategy::from_str(error_strategy.as_deref().unwrap_or("skip"))
            .ok_or_else(|| {
                Error::from_reason("invalid error_strategy; expected 'skip' or 'abort'".to_string())
            })?;
        let js_hook = JsHook {
            callback: handler,
            name: name.clone().unwrap_or_else(|| "hook".to_string()),
        };
        let config = HookConfig {
            name: name.unwrap_or_else(|| "hook".to_string()),
            error_strategy: strategy,
            handler: Arc::new(js_hook),
        };
        self.inner.hooks.add(hook_stage, config).await;
        Ok(())
    }

    #[napi(js_name = "toolsRegistryPtr")]
    pub fn tools_registry_ptr(&self) -> i64 {
        Arc::as_ptr(&self.inner.registry) as i64
    }

    #[napi(js_name = "hooksRegistryPtr")]
    pub fn hooks_registry_ptr(&self) -> i64 {
        Arc::as_ptr(&self.inner.hooks) as i64
    }

    #[napi]
    pub async fn register_guardrail(
        &self,
        handler: GuardrailCallback,
        stage: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let stage_s = stage.as_deref().unwrap_or("both");
        let stages: Vec<GuardrailStage> = match stage_s {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(Error::from_reason(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        let name_s = name.unwrap_or_else(|| "guardrail".to_string());
        let guardrails = Arc::clone(&self.inner.guardrails);
        let handler = Arc::new(handler);
        for s in &stages {
            let filter = match s {
                GuardrailStage::Input => JsGuardrailStageFilter::Input,
                GuardrailStage::Output => JsGuardrailStageFilter::Output,
            };
            let g = Arc::new(JsGuardrail {
                callback: Arc::clone(&handler),
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
        Ok(())
    }

    #[napi]
    pub async fn add_blocklist_guardrail(
        &self,
        patterns: Vec<String>,
        action: Option<String>,
        stage: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let bl_action = match action.as_deref().unwrap_or("block") {
            "block" => BlocklistAction::Block,
            "redact" => BlocklistAction::Redact,
            other => {
                return Err(Error::from_reason(format!(
                    "invalid action {other:?}; expected 'block' or 'redact'"
                )));
            }
        };
        let patterns_ref: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
        let mut handler = BlocklistGuardrail::new(&patterns_ref, bl_action)
            .map_err(|e| Error::from_reason(format!("invalid regex: {e}")))?;
        let stage_s = stage.as_deref().unwrap_or("both");
        let stages: Vec<GuardrailStage> = match stage_s {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(Error::from_reason(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        handler = handler.for_stages(stages.clone());
        let guardrails = Arc::clone(&self.inner.guardrails);
        let name_str = name.unwrap_or_else(|| "blocklist".to_string());

        if stages.contains(&GuardrailStage::Input) {
            guardrails
                .add_input(GuardrailConfig {
                    name: name_str.clone(),
                    handler: Arc::new(
                        BlocklistGuardrail::new(&patterns_ref, bl_action)
                            .map_err(|e| Error::from_reason(format!("invalid regex: {e}")))?
                            .for_stages(vec![GuardrailStage::Input]),
                    ),
                })
                .await;
        }
        if stages.contains(&GuardrailStage::Output) {
            guardrails
                .add_output(GuardrailConfig {
                    name: name_str,
                    handler: Arc::new(handler),
                })
                .await;
        }
        Ok(())
    }

    #[napi]
    pub async fn add_max_length_guardrail(
        &self,
        max_input: Option<u32>,
        max_output: Option<u32>,
        strategy: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let len_strategy = match strategy.as_deref().unwrap_or("block") {
            "block" => LengthStrategy::Block,
            "truncate" => LengthStrategy::Truncate,
            other => {
                return Err(Error::from_reason(format!(
                    "invalid strategy {other:?}; expected 'block' or 'truncate'"
                )));
            }
        };
        let guardrails = Arc::clone(&self.inner.guardrails);
        let handler: Arc<dyn GuardrailHandler> = Arc::new(MaxLengthGuardrail::new(
            max_input.map(|u| u as usize),
            max_output.map(|u| u as usize),
            len_strategy,
        ));
        let name_s = name.unwrap_or_else(|| "max_length".to_string());
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
                    handler,
                })
                .await;
        }
        Ok(())
    }

    #[napi]
    pub async fn add_pii_guardrail(
        &self,
        stage: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let stage_s = stage.as_deref().unwrap_or("both");
        let stages: Vec<GuardrailStage> = match stage_s {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(Error::from_reason(format!(
                    "invalid stage {other:?}; expected 'input', 'output', or 'both'"
                )));
            }
        };
        let guardrails = Arc::clone(&self.inner.guardrails);
        let name_s = name.unwrap_or_else(|| "pii_redact".to_string());
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
        Ok(())
    }

    #[napi]
    pub async fn run_agent(
        &self,
        spec: &AgentSpecJs,
        user_message: String,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);

        let engine = SgAgentEngine::new(spec.inner.clone())
            .with_hooks(Arc::clone(&hooks))
            .with_guardrails(Arc::clone(&guardrails));

        let mut opts = self.inner.options.clone();
        if !spec.inner.model.is_empty() {
            opts.model = spec.inner.model.clone();
        }
        opts.max_tool_rounds = spec.inner.max_tool_rounds;
        opts.request_id = request_id;
        if let Some(re) = &spec.inner.reasoning_effort {
            opts.reasoning_effort = Some(re.clone());
        }

        let outcome = engine
            .run(&http, &registry, user_message, &opts)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;
        completion_outcome_to_js(outcome)
    }

    #[napi]
    pub async fn complete(
        &self,
        user_message: String,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        let messages = vec![ChatMessage::text("user", user_message)];
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let options = finalize_call_options(&self.inner.options, request_id, reasoning_effort);

        let outcome =
            complete_with_tools(&http, &registry, &hooks, &guardrails, messages, &options)
                .await
                .map_err(|e| Error::from_reason(e.to_string()))?;
        completion_outcome_to_js(outcome)
    }

    /// gluellm-compatible alias for [`Client::complete`].
    #[napi]
    pub async fn response(
        &self,
        user_message: String,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        self.complete(
            user_message,
            timeout_secs,
            connect_timeout_secs,
            request_id,
            reasoning_effort,
        )
        .await
    }

    #[napi]
    pub async fn complete_messages(
        &self,
        messages: Value,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        let rust_messages = chat_messages_from_value(messages)?;
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let options = finalize_call_options(&self.inner.options, request_id, reasoning_effort);

        let outcome = complete_with_tools(
            &http,
            &registry,
            &hooks,
            &guardrails,
            rust_messages,
            &options,
        )
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;
        completion_outcome_to_js(outcome)
    }

    #[napi]
    pub async fn stream(
        &self,
        user_message: String,
        on_token: ThreadsafeFunction<String, ()>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<StreamOutcomeJs> {
        let messages = vec![ChatMessage::text("user", user_message)];
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let options = finalize_call_options(&self.inner.options, request_id, reasoning_effort);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let registry = Arc::clone(&self.inner.registry);
        let has_tools = !registry.list_specs().await.is_empty();
        let cb = on_token;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let consumer = tokio::task::spawn_blocking(move || {
            while let Some(delta) = rx.blocking_recv() {
                let _ = cb.call(Ok(delta), ThreadsafeFunctionCallMode::Blocking);
            }
        });

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

        let _ = consumer.await;

        let outcome = stream_result.map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(stream_outcome_to_js(outcome))
    }

    #[napi]
    pub async fn complete_response(
        &self,
        user_message: String,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<ResponseOutcomeJs> {
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let options = finalize_call_options(&self.inner.options, request_id, reasoning_effort);
        let outcome = complete_response_with_tools(
            &http,
            &registry,
            &hooks,
            &guardrails,
            user_message,
            &options,
        )
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;
        let usage = outcome.usage.map(|u| {
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            })
        });
        Ok(ResponseOutcomeJs {
            id: outcome.id,
            content: outcome.content,
            rounds: outcome.rounds,
            usage,
            request_id: outcome.request_id,
            model_used: outcome.model_used,
        })
    }

    #[napi]
    pub async fn stream_response(
        &self,
        user_message: String,
        on_token: ThreadsafeFunction<String, ()>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
    ) -> Result<ResponseStreamOutcomeJs> {
        let http = effective_http(&self.inner.http, timeout_secs, connect_timeout_secs)?;
        let mut options = self.inner.options.clone();
        options.request_id = request_id;
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let cb = on_token;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let consumer = tokio::task::spawn_blocking(move || {
            while let Some(delta) = rx.blocking_recv() {
                let _ = cb.call(Ok(delta), ThreadsafeFunctionCallMode::Blocking);
            }
        });

        let on_delta = move |delta: String| {
            let _ = tx.send(delta);
        };

        let stream_result =
            stream_response_api(&http, &hooks, &guardrails, user_message, &options, on_delta).await;

        let _ = consumer.await;
        let outcome = stream_result.map_err(|e| Error::from_reason(e.to_string()))?;
        let usage = outcome.usage.map(|u| {
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            })
        });
        Ok(ResponseStreamOutcomeJs {
            id: outcome.id,
            content: outcome.content,
            usage,
            request_id: outcome.request_id,
            model_used: outcome.model_used,
        })
    }

    #[napi]
    pub async fn connect_mcp_stdio(
        &self,
        command: String,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        prefix: Option<String>,
    ) -> Result<()> {
        let registry = Arc::clone(&self.inner.registry);
        let session = McpSession::connect_stdio(McpStdioConfig {
            command,
            args: args.unwrap_or_default(),
            env,
            label: prefix.clone(),
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;
        session
            .register_tools(&registry, prefix.as_deref())
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;
        self.inner.mcp_sessions.lock().unwrap().push(session);
        Ok(())
    }

    #[napi]
    pub async fn connect_mcp_http(&self, url: String, prefix: Option<String>) -> Result<()> {
        let registry = Arc::clone(&self.inner.registry);
        let session = McpSession::connect_http(McpHttpConfig {
            url,
            label: prefix.clone(),
            auth_header: None,
            custom_headers: Default::default(),
        })
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;
        session
            .register_tools(&registry, prefix.as_deref())
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;
        self.inner.mcp_sessions.lock().unwrap().push(session);
        Ok(())
    }

    #[napi]
    pub async fn batch(
        &self,
        requests: Vec<BatchRequestJs>,
        max_concurrent: Option<u32>,
        error_strategy: Option<String>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
    ) -> Result<BatchResponseJs> {
        let strategy = ErrorStrategy::from_str(error_strategy.as_deref().unwrap_or("continue"))
            .ok_or_else(|| {
                Error::from_reason(
                    "invalid error_strategy; expected 'continue', 'skip', or 'fail_fast'"
                        .to_string(),
                )
            })?;

        let batch_requests: Vec<BatchRequest> = requests
            .into_iter()
            .map(|r| {
                let mut br = BatchRequest::new(r.prompt);
                br.id = r.id;
                br.system_prompt = r.system_prompt;
                br
            })
            .collect();

        let config = BatchConfig {
            max_concurrent: max_concurrent.unwrap_or(5) as usize,
            error_strategy: strategy,
            timeout: timeout_secs.map(|s| Duration::from_secs(s.max(0) as u64)),
            connect_timeout: connect_timeout_secs.map(|s| Duration::from_secs(s.max(0) as u64)),
            cancel: None,
        };

        let http = Arc::clone(&self.inner.http);
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);
        let options = self.inner.options.clone();

        let response = batch_complete(
            http,
            registry,
            hooks,
            guardrails,
            batch_requests,
            &options,
            config,
        )
        .await
        .map_err(|e| Error::from_reason(e.to_string()))?;

        let results: Vec<BatchResultJs> = response
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
                BatchResultJs {
                    id: r.id,
                    success: r.success,
                    content: r.content,
                    error: r.error,
                    rounds: r.rounds,
                    usage,
                    elapsed_secs: r.elapsed_secs,
                }
            })
            .collect();

        let total_usage = response.total_usage.map(|u| {
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            })
        });

        Ok(BatchResponseJs {
            total_requests: response.total_requests as u32,
            successful: response.successful as u32,
            failed: response.failed as u32,
            elapsed_secs: response.elapsed_secs,
            total_usage,
            results,
        })
    }
}

#[napi]
pub struct Conversation {
    state: Arc<ClientState>,
    turns: std::sync::Mutex<Vec<ChatMessage>>,
}

#[napi]
impl Conversation {
    #[napi(constructor)]
    pub fn new(client: &Client) -> Self {
        let c = &client.inner;
        Self {
            state: Arc::clone(c),
            turns: std::sync::Mutex::new(Vec::new()),
        }
    }

    #[napi]
    pub fn push_user(&self, text: String) -> Result<()> {
        let mut g = self
            .turns
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        g.push(ChatMessage::text("user", text));
        Ok(())
    }

    #[napi]
    pub fn push_assistant_text(&self, text: String) -> Result<()> {
        let mut g = self
            .turns
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        g.push(ChatMessage::text("assistant", text));
        Ok(())
    }

    #[napi(getter)]
    pub fn messages(&self) -> Result<Value> {
        let g = self
            .turns
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        serde_json::to_value(&*g).map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub async fn complete(
        &self,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
        reasoning_effort: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        let msgs = self
            .turns
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?
            .clone();
        let http = effective_http(&self.state.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.state.registry);
        let hooks = Arc::clone(&self.state.hooks);
        let guardrails = Arc::clone(&self.state.guardrails);
        let options = finalize_call_options(&self.state.options, request_id, reasoning_effort);

        let outcome = complete_with_tools(&http, &registry, &hooks, &guardrails, msgs, &options)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        {
            let mut g = self
                .turns
                .lock()
                .map_err(|e| Error::from_reason(e.to_string()))?;
            *g = outcome.messages.clone();
        }
        completion_outcome_to_js(outcome)
    }
}

#[napi(js_name = "AgentEngine")]
#[derive(Clone)]
pub struct JsAgentEngine {
    spec: AgentSpec,
    base_options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
}

#[napi]
impl JsAgentEngine {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: &AgentSpecJs,
        api_key: String,
        base_url: Option<String>,
        model: Option<String>,
        max_retries: Option<u32>,
        retry_initial_delay_ms: Option<u32>,
        retry_max_delay_ms: Option<u32>,
        retry_multiplier: Option<f64>,
        requests_per_second: Option<u32>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        pool_max_idle_per_host: Option<u32>,
        pool_idle_timeout_secs: Option<i64>,
        reasoning_effort: Option<String>,
        status_emitter: Option<&StatusEmitterJs>,
    ) -> Result<Self> {
        let max_retries = max_retries.unwrap_or(3);
        let cfg = ClientConfig {
            retry: RetryPolicy {
                max_retries,
                initial_interval_ms: u64::from(retry_initial_delay_ms.unwrap_or(50)),
                max_interval_ms: u64::from(retry_max_delay_ms.unwrap_or(2000)),
                multiplier: retry_multiplier.unwrap_or(2.0),
            },
            quota_per_second: requests_per_second.and_then(NonZeroU32::new),
            timeout: Duration::from_secs(timeout_secs.unwrap_or(60).max(0) as u64),
            connect_timeout: Duration::from_secs(connect_timeout_secs.unwrap_or(30).max(0) as u64),
            pool_max_idle_per_host: pool_max_idle_per_host.unwrap_or(50) as usize,
            pool_idle_timeout: pool_idle_timeout_secs.map(|s| Duration::from_secs(s.max(0) as u64)),
            ..ClientConfig::default()
        };
        let http = HttpClient::new(cfg).map_err(|e| Error::from_reason(e.to_string()))?;

        let effective_model = if spec.inner.model.is_empty() {
            model.unwrap_or_else(|| "gpt-5.4-nano-2026-03-17-mini".to_string())
        } else {
            spec.inner.model.clone()
        };

        let guardrails = Arc::new(
            GuardrailRegistry::new().with_max_output_retries(spec.inner.max_output_retries),
        );

        let status_emitter_arc = status_emitter.map(|e| Arc::clone(&e.inner));
        let effective_reasoning = reasoning_effort.or_else(|| spec.inner.reasoning_effort.clone());
        let mut base_options = ChatOptions::new(
            base_url.as_deref().unwrap_or("https://api.openai.com"),
            api_key,
            effective_model,
        );
        base_options.reasoning_effort = effective_reasoning;
        base_options.status_emitter = status_emitter_arc;

        Ok(Self {
            spec: spec.inner.clone(),
            base_options,
            registry: Arc::new(ToolRegistry::new()),
            hooks: Arc::new(HookRegistry::new()),
            guardrails,
            http: Arc::new(http),
        })
    }

    #[napi]
    pub async fn register_tool(
        &self,
        name: String,
        description: String,
        parameters: Value,
        callback: JsonCallback,
        static_tool: Option<bool>,
    ) -> Result<()> {
        let spec = ToolSpec {
            name,
            description: Some(description),
            parameters_schema: parameters,
            static_tool: static_tool.unwrap_or(false),
        };
        let tool = Arc::new(JsDictTool { spec, callback }) as Arc<dyn Tool>;
        self.registry
            .register(tool)
            .await
            .map_err(tool_error_to_napi)
    }

    #[napi]
    pub async fn register_hook(
        &self,
        stage: String,
        handler: JsonCallback,
        name: Option<String>,
        error_strategy: Option<String>,
    ) -> Result<()> {
        let hook_stage = HookStage::from_str(&stage)
            .ok_or_else(|| Error::from_reason(format!("invalid stage {stage:?}")))?;
        let strategy = HookErrorStrategy::from_str(error_strategy.as_deref().unwrap_or("skip"))
            .ok_or_else(|| Error::from_reason("invalid error_strategy".to_string()))?;
        let js_hook = JsHook {
            callback: handler,
            name: name.clone().unwrap_or_else(|| "hook".to_string()),
        };
        let config = HookConfig {
            name: name.unwrap_or_else(|| "hook".to_string()),
            error_strategy: strategy,
            handler: Arc::new(js_hook),
        };
        self.hooks.add(hook_stage, config).await;
        Ok(())
    }

    #[napi]
    pub async fn register_guardrail(
        &self,
        handler: GuardrailCallback,
        stage: Option<String>,
        name: Option<String>,
    ) -> Result<()> {
        let stage_s = stage.as_deref().unwrap_or("both");
        let stages: Vec<GuardrailStage> = match stage_s {
            "input" => vec![GuardrailStage::Input],
            "output" => vec![GuardrailStage::Output],
            "both" => vec![GuardrailStage::Input, GuardrailStage::Output],
            other => {
                return Err(Error::from_reason(format!("invalid stage {other:?}")));
            }
        };
        let name_s = name.unwrap_or_else(|| "guardrail".to_string());
        let handler = Arc::new(handler);
        for s in &stages {
            let filter = match s {
                GuardrailStage::Input => JsGuardrailStageFilter::Input,
                GuardrailStage::Output => JsGuardrailStageFilter::Output,
            };
            let g = Arc::new(JsGuardrail {
                callback: Arc::clone(&handler),
                stage_filter: filter,
                name: name_s.clone(),
            }) as Arc<dyn GuardrailHandler>;
            let config = GuardrailConfig {
                name: name_s.clone(),
                handler: g,
            };
            match s {
                GuardrailStage::Input => self.guardrails.add_input(config).await,
                GuardrailStage::Output => self.guardrails.add_output(config).await,
            }
        }
        Ok(())
    }

    #[napi]
    pub async fn run(
        &self,
        user_message: String,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
    ) -> Result<CompletionOutcomeJs> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let registry = Arc::clone(&self.registry);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);

        let engine = SgAgentEngine::new(self.spec.clone())
            .with_hooks(hooks)
            .with_guardrails(guardrails);

        let opts = finalize_call_options(&self.base_options, request_id, None);

        let outcome = engine
            .run(&http, &registry, user_message, &opts)
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;
        completion_outcome_to_js(outcome)
    }

    #[napi]
    pub async fn stream(
        &self,
        user_message: String,
        on_token: ThreadsafeFunction<String, ()>,
        timeout_secs: Option<i64>,
        connect_timeout_secs: Option<i64>,
        request_id: Option<String>,
    ) -> Result<StreamOutcomeJs> {
        let http = effective_http(&self.http, timeout_secs, connect_timeout_secs)?;
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);
        let opts = finalize_call_options(&self.base_options, request_id, None);
        let spec = self.spec.clone();
        let cb = on_token;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let consumer = tokio::task::spawn_blocking(move || {
            while let Some(delta) = rx.blocking_recv() {
                let _ = cb.call(Ok(delta), ThreadsafeFunctionCallMode::Blocking);
            }
        });

        let on_delta = move |delta: String| {
            let _ = tx.send(delta);
        };

        let engine = SgAgentEngine::new(spec)
            .with_hooks(hooks)
            .with_guardrails(guardrails);
        let stream_result = engine.stream(&http, user_message, &opts, on_delta).await;
        let _ = consumer.await;

        let outcome = stream_result.map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(stream_outcome_to_js(outcome))
    }
}

#[napi(js_name = "version")]
pub fn version_string() -> &'static str {
    superglue::version()
}
