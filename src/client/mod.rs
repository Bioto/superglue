//! High-level [`Client`] — bundles HTTP, chat options, tools, hooks, and guardrails.
//!
//! Mirrors the `Client` type exposed by Python, JavaScript, and Kotlin bindings.

mod bootstrap;

use std::collections::HashMap;
use std::sync::Arc;

use std::time::Duration;
use thiserror::Error;
#[cfg(feature = "mcp")]
use tokio::sync::Mutex;

use crate::agents::{AgentEngine, AgentSpec};
use crate::audit::{RunRecorder, RunStore, now_ms};
use crate::batch::{BatchConfig, BatchError, BatchRequest, BatchResponse, batch_complete};
use crate::chat::{
    ChatError, ChatOptions, CompletionOutcome, Conversation, StreamOutcome, complete_with_tools,
    stream_complete, stream_complete_with_tools,
};
use crate::context::SummarizeContextConfig;
use crate::events::{StatusEmitter, StatusSubscriber};
use crate::fallback::{FallbackPolicy, ModelFallbackChain};
use crate::guardrails::{
    BlocklistAction, BlocklistGuardrail, GuardrailConfig, GuardrailHandler, GuardrailRegistry,
    GuardrailStage, LengthStrategy, MaxLengthGuardrail, PiiRedactGuardrail,
};
use crate::hooks::{HookConfig, HookErrorStrategy, HookRegistry, HookStage};
use crate::http::{Error as HttpError, HttpClient};
use crate::openai::ChatMessage;
use crate::responses::{
    ResponseError, ResponseOutcome, ResponseStreamOutcome,
    complete_with_tools as complete_response_with_tools, stream_response as stream_response_api,
};
use crate::tools::{Tool, ToolMode, ToolRegistry};

pub use bootstrap::{
    BindingBootstrap, BindingBootstrapConfig, bootstrap_from_parts, provider_id_from_str,
};

const DEFAULT_MODEL: &str = "gpt-5.4-nano-2026-03-17-mini";
const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Errors from [`ClientBuilder::build`] and [`Client::from_env`].
#[derive(Debug, Error)]
pub enum ClientBuildError {
    #[error("api_key is required")]
    MissingApiKey,
    #[error("missing API key for provider {0}")]
    MissingProviderKey(crate::providers::ProviderId),
    #[error(transparent)]
    Http(#[from] HttpError),
}

/// Per-call overrides for [`Client::complete`], [`Client::stream`], etc.
#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    pub request_id: Option<String>,
    pub reasoning_effort: Option<String>,
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
}

struct ClientAudit {
    recorder: Arc<RunRecorder>,
    ready: tokio::sync::OnceCell<()>,
}

struct ClientInner {
    options: ChatOptions,
    registry: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
    http: Arc<HttpClient>,
    max_upload_bytes: usize,
    audit: Option<ClientAudit>,
    #[cfg(feature = "mcp")]
    mcp_sessions: Mutex<Vec<Arc<crate::mcp::McpSession>>>,
}

/// Bundled superglue client for completions, streaming, batch, tools, hooks, and agents.
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

/// Builder for [`Client`] with binding-aligned defaults.
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    api_key: Option<String>,
    model: String,
    base_url: String,
    system_prompt: Option<String>,
    max_tool_rounds: u32,
    max_retries: u32,
    retry_initial_delay_ms: u64,
    retry_max_delay_ms: u64,
    retry_multiplier: f64,
    requests_per_second: Option<u32>,
    timeout: Duration,
    connect_timeout: Duration,
    max_output_retries: u32,
    pool_max_idle_per_host: usize,
    pool_idle_timeout: Option<Duration>,
    reasoning_effort: Option<String>,
    status_emitter: Option<Arc<StatusEmitter>>,
    model_fallback: Option<ModelFallbackChain>,
    provider_credentials: Option<crate::providers::ProviderCredentials>,
    provider_qps: HashMap<crate::providers::ProviderId, u32>,
    max_upload_bytes: usize,
    tool_mode: ToolMode,
    tool_route_model: Option<String>,
    condense_tool_messages: bool,
    aaak_tool_condensing: bool,
    summarize_context: SummarizeContextConfig,
    aaak_compression_enabled: bool,
    aaak_compression_model: Option<String>,
    audit_store: Option<Arc<RunStore>>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder {
            api_key: None,
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            system_prompt: None,
            max_tool_rounds: 16,
            max_retries: 3,
            retry_initial_delay_ms: 50,
            retry_max_delay_ms: 2000,
            retry_multiplier: 2.0,
            requests_per_second: None,
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
            max_output_retries: 3,
            pool_max_idle_per_host: 50,
            pool_idle_timeout: None,
            reasoning_effort: None,
            status_emitter: None,
            model_fallback: None,
            provider_credentials: None,
            provider_qps: HashMap::new(),
            max_upload_bytes: crate::files::default_max_upload_bytes(),
            tool_mode: ToolMode::default(),
            tool_route_model: None,
            condense_tool_messages: false,
            aaak_tool_condensing: false,
            summarize_context: SummarizeContextConfig::default(),
            aaak_compression_enabled: false,
            aaak_compression_model: None,
            audit_store: None,
        }
    }
}

impl ClientBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn max_tool_rounds(mut self, n: u32) -> Self {
        self.max_tool_rounds = n;
        self
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn retry_initial_delay_ms(mut self, ms: u64) -> Self {
        self.retry_initial_delay_ms = ms;
        self
    }

    pub fn retry_max_delay_ms(mut self, ms: u64) -> Self {
        self.retry_max_delay_ms = ms;
        self
    }

    pub fn retry_multiplier(mut self, m: f64) -> Self {
        self.retry_multiplier = m;
        self
    }

    /// Set API key for a specific provider (overrides env for that provider).
    pub fn api_key_for(
        mut self,
        provider: crate::providers::ProviderId,
        key: impl Into<String>,
    ) -> Self {
        let mut creds = self.provider_credentials.unwrap_or_else(|| {
            let mut c = crate::providers::ProviderCredentials::from_env();
            c.with_legacy_openai_key(self.api_key.as_deref().unwrap_or(""), Some(&self.base_url));
            c
        });
        creds.insert_key(provider, key);
        self.provider_credentials = Some(creds);
        self
    }

    /// Per-provider default QPS when a new API-key bucket is created.
    pub fn requests_per_second_for(
        mut self,
        provider: crate::providers::ProviderId,
        qps: u32,
    ) -> Self {
        self.provider_qps.insert(provider, qps);
        self
    }

    pub fn max_upload_bytes(mut self, bytes: usize) -> Self {
        self.max_upload_bytes = bytes;
        self
    }

    pub fn requests_per_second(mut self, qps: u32) -> Self {
        self.requests_per_second = Some(qps);
        self
    }

    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    pub fn connect_timeout(mut self, d: Duration) -> Self {
        self.connect_timeout = d;
        self
    }

    pub fn max_output_retries(mut self, n: u32) -> Self {
        self.max_output_retries = n;
        self
    }

    pub fn pool_max_idle_per_host(mut self, n: usize) -> Self {
        self.pool_max_idle_per_host = n;
        self
    }

    pub fn pool_idle_timeout(mut self, d: Duration) -> Self {
        self.pool_idle_timeout = Some(d);
        self
    }

    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn status_emitter(mut self, emitter: Arc<StatusEmitter>) -> Self {
        self.status_emitter = Some(emitter);
        self
    }

    /// Dynamic tool routing mode (`standard` or `dynamic`; GlueLLM parity).
    pub fn tool_mode(mut self, mode: ToolMode) -> Self {
        self.tool_mode = mode;
        self
    }

    pub fn tool_route_model(mut self, model: impl Into<String>) -> Self {
        self.tool_route_model = Some(model.into());
        self
    }

    pub fn condense_tool_messages(mut self, enabled: bool) -> Self {
        self.condense_tool_messages = enabled;
        self
    }

    pub fn aaak_tool_condensing(mut self, enabled: bool) -> Self {
        self.aaak_tool_condensing = enabled;
        self
    }

    pub fn summarize_context(mut self, config: SummarizeContextConfig) -> Self {
        self.summarize_context = config;
        self
    }

    pub fn aaak_compression_enabled(mut self, enabled: bool) -> Self {
        self.aaak_compression_enabled = enabled;
        self
    }

    pub fn aaak_compression_model(mut self, model: impl Into<String>) -> Self {
        self.aaak_compression_model = Some(model.into());
        self
    }

    /// Ordered model fallback chain (primary first). Applied after per-request HTTP retries.
    pub fn model_fallback_chain(
        mut self,
        models: Vec<String>,
        policy: Option<FallbackPolicy>,
    ) -> Self {
        let chain = match policy {
            Some(p) => ModelFallbackChain::new(models).with_policy(p),
            None => ModelFallbackChain::new(models),
        };
        self.model_fallback = Some(chain);
        self
    }

    /// Attach an in-memory audit store; completed chat and Responses turns are recorded automatically.
    pub fn audit(mut self, store: Arc<RunStore>) -> Self {
        self.audit_store = Some(store);
        self
    }

    /// Build a [`Client`].
    ///
    /// # Errors
    ///
    /// Returns [`ClientBuildError`] if `api_key` is missing or the HTTP client fails to construct.
    pub fn build(self) -> Result<Client, ClientBuildError> {
        let bootstrap = bootstrap_from_parts(BindingBootstrapConfig {
            api_key: self
                .api_key
                .filter(|k| !k.is_empty())
                .ok_or(ClientBuildError::MissingApiKey)?,
            base_url: self.base_url,
            model: self.model,
            system_prompt: self.system_prompt,
            max_tool_rounds: self.max_tool_rounds,
            max_retries: self.max_retries,
            retry_initial_delay_ms: self.retry_initial_delay_ms,
            retry_max_delay_ms: self.retry_max_delay_ms,
            retry_multiplier: self.retry_multiplier,
            requests_per_second: self.requests_per_second,
            timeout: self.timeout,
            connect_timeout: self.connect_timeout,
            max_output_retries: self.max_output_retries,
            pool_max_idle_per_host: self.pool_max_idle_per_host,
            pool_idle_timeout: self.pool_idle_timeout,
            reasoning_effort: self.reasoning_effort,
            status_emitter: self.status_emitter,
            model_fallback: self.model_fallback,
            provider_credentials: self.provider_credentials,
            provider_qps: self.provider_qps,
            max_upload_bytes: self.max_upload_bytes,
        })?;
        let max_output_retries = self.max_output_retries;
        let mut options = bootstrap.options;
        options.tool_mode = self.tool_mode;
        options.tool_route_model = self.tool_route_model;
        options.condense_tool_messages = self.condense_tool_messages;
        options.aaak_tool_condensing = self.aaak_tool_condensing;
        options.summarize_context = self.summarize_context;
        options.aaak_compression_enabled = self.aaak_compression_enabled;
        options.aaak_compression_model = self.aaak_compression_model;

        let audit = self.audit_store.map(|store| ClientAudit {
            recorder: Arc::new(RunRecorder::new(store)),
            ready: tokio::sync::OnceCell::new(),
        });

        Ok(Client {
            inner: Arc::new(ClientInner {
                options,
                registry: Arc::new(ToolRegistry::new()),
                hooks: Arc::new(HookRegistry::new()),
                guardrails: Arc::new(
                    GuardrailRegistry::new().with_max_output_retries(max_output_retries),
                ),
                http: Arc::new(bootstrap.http),
                max_upload_bytes: bootstrap.max_upload_bytes,
                audit,
                #[cfg(feature = "mcp")]
                mcp_sessions: Mutex::new(Vec::new()),
            }),
        })
    }
}

impl Client {
    #[must_use]
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Build from `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientBuildError`] if the API key is unset or the client fails to construct.
    pub fn from_env() -> Result<Self, ClientBuildError> {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let base_url =
            std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        Client::builder()
            .api_key(api_key)
            .model(model)
            .base_url(base_url)
            .build()
    }

    /// Register a tool implementation.
    pub async fn register_tool(
        &self,
        tool: Arc<dyn Tool>,
    ) -> Result<(), crate::tools::ToolInvokeError> {
        self.inner.registry.register(tool).await
    }

    /// Register a lifecycle hook.
    pub async fn register_hook(&self, stage: HookStage, config: HookConfig) {
        self.inner.hooks.add(stage, config).await;
    }

    /// Register a custom guardrail on input and/or output stages.
    pub async fn register_guardrail(
        &self,
        stages: &[GuardrailStage],
        name: impl Into<String>,
        handler: Arc<dyn GuardrailHandler>,
    ) {
        let name = name.into();
        let guardrails = Arc::clone(&self.inner.guardrails);
        for stage in stages {
            let config = GuardrailConfig {
                name: name.clone(),
                handler: Arc::clone(&handler),
            };
            match stage {
                GuardrailStage::Input => guardrails.add_input(config).await,
                GuardrailStage::Output => guardrails.add_output(config).await,
            }
        }
    }

    /// Built-in regex blocklist guardrail.
    pub async fn add_blocklist_guardrail(
        &self,
        patterns: &[String],
        action: BlocklistAction,
        stages: &[GuardrailStage],
        name: impl Into<String>,
    ) -> Result<(), regex::Error> {
        let patterns_ref: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
        let name = name.into();
        let guardrails = Arc::clone(&self.inner.guardrails);

        if stages.contains(&GuardrailStage::Input) {
            guardrails
                .add_input(GuardrailConfig {
                    name: name.clone(),
                    handler: Arc::new(
                        BlocklistGuardrail::new(&patterns_ref, action)?
                            .for_stages(vec![GuardrailStage::Input]),
                    ),
                })
                .await;
        }
        if stages.contains(&GuardrailStage::Output) {
            guardrails
                .add_output(GuardrailConfig {
                    name,
                    handler: Arc::new(
                        BlocklistGuardrail::new(&patterns_ref, action)?
                            .for_stages(vec![GuardrailStage::Output]),
                    ),
                })
                .await;
        }
        Ok(())
    }

    /// Built-in max-length guardrail.
    pub async fn add_max_length_guardrail(
        &self,
        max_input: Option<usize>,
        max_output: Option<usize>,
        strategy: LengthStrategy,
        name: impl Into<String>,
    ) {
        let name = name.into();
        let handler: Arc<dyn GuardrailHandler> =
            Arc::new(MaxLengthGuardrail::new(max_input, max_output, strategy));
        let guardrails = Arc::clone(&self.inner.guardrails);
        if max_input.is_some() {
            guardrails
                .add_input(GuardrailConfig {
                    name: name.clone(),
                    handler: Arc::clone(&handler),
                })
                .await;
        }
        if max_output.is_some() {
            guardrails
                .add_output(GuardrailConfig { name, handler })
                .await;
        }
    }

    /// Built-in PII redaction guardrail.
    pub async fn add_pii_guardrail(&self, stages: &[GuardrailStage], name: impl Into<String>) {
        let name = name.into();
        let guardrails = Arc::clone(&self.inner.guardrails);
        if stages.contains(&GuardrailStage::Input) {
            guardrails
                .add_input(GuardrailConfig {
                    name: name.clone(),
                    handler: Arc::new(
                        PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Input]),
                    ),
                })
                .await;
        }
        if stages.contains(&GuardrailStage::Output) {
            guardrails
                .add_output(GuardrailConfig {
                    name,
                    handler: Arc::new(
                        PiiRedactGuardrail::new().for_stages(vec![GuardrailStage::Output]),
                    ),
                })
                .await;
        }
    }

    /// Single-turn completion with an optional user message.
    pub async fn complete(
        &self,
        user_message: impl Into<String>,
        call: CallOptions,
    ) -> Result<CompletionOutcome, ChatError> {
        let messages = vec![ChatMessage::text("user", user_message.into())];
        self.complete_messages(messages, call).await
    }

    /// Completion with a caller-supplied message list.
    pub async fn complete_messages(
        &self,
        messages: Vec<ChatMessage>,
        call: CallOptions,
    ) -> Result<CompletionOutcome, ChatError> {
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let options = finalize_call_options(&self.inner.options, &call);
        self.ensure_audit_ready().await;
        let started_at_ms = now_ms();
        let result = complete_with_tools(
            &http,
            &self.inner.registry,
            &self.inner.hooks,
            &self.inner.guardrails,
            messages,
            &options,
        )
        .await;
        if let (Ok(outcome), Some(audit)) = (&result, self.inner.audit.as_ref()) {
            audit
                .recorder
                .record_chat_completion(outcome, &options, started_at_ms)
                .await;
        }
        result
    }

    /// Stream a completion; `on_delta` receives each content token.
    pub async fn stream<F>(
        &self,
        user_message: impl Into<String>,
        call: CallOptions,
        on_delta: F,
    ) -> Result<StreamOutcome, ChatError>
    where
        F: FnMut(String) + Send,
    {
        let messages = vec![ChatMessage::text("user", user_message.into())];
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let options = finalize_call_options(&self.inner.options, &call);
        let specs = self.inner.registry.list_specs().await;
        if specs.is_empty() {
            stream_complete(
                &http,
                &self.inner.hooks,
                &self.inner.guardrails,
                messages,
                &options,
                on_delta,
            )
            .await
        } else {
            let out = stream_complete_with_tools(
                &http,
                &self.inner.registry,
                &self.inner.hooks,
                &self.inner.guardrails,
                messages,
                &options,
                on_delta,
            )
            .await?;
            Ok(StreamOutcome {
                content: out.content,
                finish_reason: out.finish_reason,
                usage: out.usage,
                request_id: out.request_id,
            })
        }
    }

    /// Upload a file to the provider Files API (or prepare inline metadata for Anthropic).
    pub async fn upload_file(
        &self,
        path: impl AsRef<std::path::Path>,
        purpose: crate::files::FilePurpose,
        provider: Option<crate::providers::ProviderId>,
    ) -> Result<crate::files::UploadedFile, crate::files::FileError> {
        let creds = crate::chat::credentials_for(&self.inner.options);
        let provider = provider.unwrap_or_else(|| {
            crate::providers::parse_model_ref(&self.inner.options.model).provider
        });
        crate::files::upload_file(
            &self.inner.http,
            &creds,
            provider,
            path.as_ref(),
            purpose,
            self.inner.max_upload_bytes,
        )
        .await
    }

    /// Build a user message with inline file bytes for chat.
    #[must_use]
    pub fn message_with_file_bytes(
        text: Option<&str>,
        filename: &str,
        bytes: &[u8],
    ) -> ChatMessage {
        crate::files::message_with_file_bytes(text, filename, bytes)
    }

    /// Non-streaming completion via the OpenAI Responses API (`/v1/responses`).
    pub async fn complete_response(
        &self,
        user_message: impl Into<String>,
        call: CallOptions,
    ) -> Result<ResponseOutcome, ResponseError> {
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let options = finalize_call_options(&self.inner.options, &call);
        self.ensure_audit_ready().await;
        let started_at_ms = now_ms();
        let result = complete_response_with_tools(
            &http,
            &self.inner.registry,
            &self.inner.hooks,
            &self.inner.guardrails,
            user_message,
            &options,
        )
        .await;
        if let (Ok(outcome), Some(audit)) = (&result, self.inner.audit.as_ref()) {
            audit
                .recorder
                .record_response_completion(outcome, &options, started_at_ms)
                .await;
        }
        result
    }

    /// Stream a Responses API completion; `on_delta` receives text token deltas.
    pub async fn stream_response<F>(
        &self,
        user_message: impl Into<String>,
        call: CallOptions,
        on_delta: F,
    ) -> Result<ResponseStreamOutcome, ResponseError>
    where
        F: FnMut(String) + Send,
    {
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let options = finalize_call_options(&self.inner.options, &call);
        stream_response_api(
            &http,
            &self.inner.hooks,
            &self.inner.guardrails,
            user_message,
            &options,
            on_delta,
        )
        .await
    }

    #[cfg(feature = "mcp")]
    /// Connect an MCP server over stdio and register its tools into this client's registry.
    pub async fn connect_mcp_stdio(
        &self,
        command: impl Into<String>,
        args: Vec<String>,
        env: Option<std::collections::HashMap<String, String>>,
        prefix: Option<String>,
    ) -> Result<Arc<crate::mcp::McpSession>, crate::mcp::McpError> {
        let session = crate::mcp::McpSession::connect_stdio(crate::mcp::McpStdioConfig {
            command: command.into(),
            args,
            env,
            label: prefix.clone(),
        })
        .await?;
        session
            .register_tools(&self.inner.registry, prefix.as_deref())
            .await?;
        self.inner
            .mcp_sessions
            .lock()
            .await
            .push(Arc::clone(&session));
        Ok(session)
    }

    #[cfg(feature = "mcp")]
    /// Connect an MCP server over streamable HTTP and register its tools.
    pub async fn connect_mcp_http(
        &self,
        url: impl Into<String>,
        prefix: Option<String>,
    ) -> Result<Arc<crate::mcp::McpSession>, crate::mcp::McpError> {
        let session = crate::mcp::McpSession::connect_http(crate::mcp::McpHttpConfig {
            url: url.into(),
            label: prefix.clone(),
            auth_header: None,
            custom_headers: HashMap::new(),
        })
        .await?;
        session
            .register_tools(&self.inner.registry, prefix.as_deref())
            .await?;
        self.inner
            .mcp_sessions
            .lock()
            .await
            .push(Arc::clone(&session));
        Ok(session)
    }

    #[cfg(feature = "mcp")]
    /// Close an MCP session and remove it from the client's tracked list.
    pub async fn close_mcp(
        &self,
        session: Arc<crate::mcp::McpSession>,
    ) -> Result<(), crate::mcp::McpError> {
        session.close().await?;
        self.inner
            .mcp_sessions
            .lock()
            .await
            .retain(|s| !Arc::ptr_eq(s, &session));
        Ok(())
    }

    /// Run many prompts concurrently.
    pub async fn batch(
        &self,
        requests: Vec<BatchRequest>,
        config: BatchConfig,
    ) -> Result<BatchResponse, BatchError> {
        let http = if config.timeout.is_some() || config.connect_timeout.is_some() {
            effective_http(&self.inner.http, config.timeout, config.connect_timeout)
                .map_err(|e| BatchError::Internal(e.to_string()))?
        } else {
            Arc::clone(&self.inner.http)
        };
        batch_complete(
            http,
            Arc::clone(&self.inner.registry),
            Arc::clone(&self.inner.hooks),
            Arc::clone(&self.inner.guardrails),
            requests,
            &self.inner.options,
            config,
        )
        .await
    }

    /// Run an agent spec against a user message.
    pub async fn run_agent(
        &self,
        spec: AgentSpec,
        user_message: impl Into<String>,
        call: CallOptions,
    ) -> Result<CompletionOutcome, ChatError> {
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let registry = Arc::clone(&self.inner.registry);
        let hooks = Arc::clone(&self.inner.hooks);
        let guardrails = Arc::clone(&self.inner.guardrails);

        let mut opts = finalize_call_options(&self.inner.options, &call);
        if !spec.model.is_empty() {
            opts.model = spec.model.clone();
        }
        opts.max_tool_rounds = spec.max_tool_rounds;
        if let Some(re) = &spec.reasoning_effort {
            opts.reasoning_effort = Some(re.clone());
        }
        opts.request_id = call.request_id;

        self.ensure_audit_ready().await;
        let started_at_ms = now_ms();

        let engine = AgentEngine::new(spec)
            .with_hooks(hooks)
            .with_guardrails(guardrails);

        let result = engine.run(&http, &registry, user_message, &opts).await;
        if let (Ok(outcome), Some(audit)) = (&result, self.inner.audit.as_ref()) {
            audit
                .recorder
                .record_chat_completion(outcome, &opts, started_at_ms)
                .await;
        }
        result
    }

    /// Start a multi-turn conversation backed by this client.
    #[must_use]
    pub fn conversation(&self) -> ClientConversation {
        ClientConversation {
            inner: Arc::clone(&self.inner),
            conversation: Conversation::new(),
        }
    }

    async fn ensure_audit_ready(&self) {
        ensure_client_audit_ready(&self.inner).await;
    }
}

/// Multi-turn chat using a shared [`Client`] registry and options.
pub struct ClientConversation {
    inner: Arc<ClientInner>,
    conversation: Conversation,
}

impl ClientConversation {
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.conversation.push_user(text);
    }

    pub async fn complete(&mut self, call: CallOptions) -> Result<CompletionOutcome, ChatError> {
        let http = effective_http(&self.inner.http, call.timeout, call.connect_timeout)?;
        let options = finalize_call_options(&self.inner.options, &call);
        ensure_client_audit_ready(&self.inner).await;
        let started_at_ms = now_ms();
        let result = self
            .conversation
            .complete(
                &http,
                &self.inner.registry,
                &self.inner.hooks,
                &self.inner.guardrails,
                &options,
            )
            .await;
        if let (Ok(outcome), Some(audit)) = (&result, self.inner.audit.as_ref()) {
            audit
                .recorder
                .record_chat_completion(outcome, &options, started_at_ms)
                .await;
        }
        result
    }
}

fn finalize_call_options(base: &ChatOptions, call: &CallOptions) -> ChatOptions {
    let mut options = base.clone();
    if let Some(id) = &call.request_id {
        options.request_id = Some(id.clone());
    }
    if let Some(re) = &call.reasoning_effort {
        options.reasoning_effort = Some(re.clone());
    }
    options
}

async fn ensure_client_audit_ready(inner: &ClientInner) {
    let Some(audit) = inner.audit.as_ref() else {
        return;
    };
    audit
        .ready
        .get_or_init(|| async {
            for stage in RunRecorder::all_stages() {
                inner
                    .hooks
                    .add(
                        stage,
                        HookConfig {
                            name: "audit".into(),
                            error_strategy: HookErrorStrategy::Skip,
                            handler: Arc::clone(&audit.recorder)
                                as Arc<dyn crate::hooks::HookHandler>,
                        },
                    )
                    .await;
            }
            if let Some(emitter) = &inner.options.status_emitter {
                emitter
                    .subscribe(Arc::clone(&audit.recorder) as Arc<dyn StatusSubscriber>)
                    .await;
            }
        })
        .await;
}

fn effective_http(
    base: &Arc<HttpClient>,
    timeout: Option<Duration>,
    connect_timeout: Option<Duration>,
) -> Result<Arc<HttpClient>, HttpError> {
    if timeout.is_none() && connect_timeout.is_none() {
        return Ok(Arc::clone(base));
    }
    let t = timeout.unwrap_or(base.config.timeout);
    let ct = connect_timeout.unwrap_or(base.config.connect_timeout);
    Ok(Arc::new(base.clone_with_timeouts(t, ct)?))
}
