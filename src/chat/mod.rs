//! Chat completions: non-streaming tool-loop and streaming (SSE) variants.

mod context_ops;
mod conversation;
mod file_locks;
mod loop_guard;
pub mod reasoning;
pub(crate) mod stream_tools;
mod tool_summary;

pub(crate) use loop_guard::{SharedToolLoopGuard, new_tool_loop_guard};

pub(crate) use context_ops::{
    condense_tool_round, estimate_context_chars, maybe_summarize_messages,
};

use std::fmt;
use std::sync::Arc;

pub use conversation::Conversation;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use secrecy::ExposeSecret;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::pin::Pin;
use thiserror::Error;
use tokio::time::{Duration, sleep};
use tracing::instrument;

use std::time::Instant;

use self::context_ops::{resolve_tool_route, user_context_for_route};
use crate::cancel::CancellationToken;
use crate::context::SummarizeContextConfig;
use crate::costing::estimate_model_call_cost_usd;
use crate::events::{ProcessEvent, ProcessEventKind, StatusEmitter, emit_safe};
use crate::guardrails::{GuardrailError, GuardrailOutcome, GuardrailRegistry, GuardrailStage};
use crate::hooks::{HookContext, HookError, HookRegistry, HookStage};
use crate::http::{Error as HttpError, HttpClient, sse::SseParser};
use crate::openai::{
    ChatCompletionChunk, ChatMessage, MessageContent, ResponseFormat, StopSequence, ToolCall,
    ToolChoice,
};
use crate::proto;
use crate::tools::{
    ActiveToolSet, DEFAULT_TOOL_ROUTE_MODEL, OnToolError, ToolInvokeError, ToolMode, ToolRegistry,
    ToolRetryPolicy, is_router_call, router_query_from_calls,
};

/// Callback for a compact context block appended after a condensed tool round.
#[derive(Clone)]
pub struct ContextBlockProvider(Arc<dyn Fn() -> String + Send + Sync>);

impl ContextBlockProvider {
    pub fn new(provider: impl Fn() -> String + Send + Sync + 'static) -> Self {
        Self(Arc::new(provider))
    }

    pub fn render(&self) -> String {
        (self.0)()
    }
}

impl fmt::Debug for ContextBlockProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ContextBlockProvider(..)")
    }
}

/// Callback for low-volume context lifecycle diagnostics.
#[derive(Clone)]
pub struct ContextEventLogger(Arc<dyn Fn(String) + Send + Sync>);

impl ContextEventLogger {
    pub fn new(logger: impl Fn(String) + Send + Sync + 'static) -> Self {
        Self(Arc::new(logger))
    }

    pub fn log(&self, event: impl Into<String>) {
        (self.0)(event.into());
    }
}

impl fmt::Debug for ContextEventLogger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ContextEventLogger(..)")
    }
}

/// Snapshot of the full LLM request payload immediately before an HTTP completion call.
#[derive(Debug, Clone)]
pub struct LlmPayloadSnapshot {
    pub round: u32,
    pub request_id: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<ChatMessage>,
}

/// Opt-in observer for diagnostics / forensics (system prompt + messages per round).
#[derive(Clone)]
pub struct LlmPayloadObserver(Arc<dyn Fn(LlmPayloadSnapshot) + Send + Sync>);

impl LlmPayloadObserver {
    pub fn new(observer: impl Fn(LlmPayloadSnapshot) + Send + Sync + 'static) -> Self {
        Self(Arc::new(observer))
    }

    pub fn notify(&self, snapshot: LlmPayloadSnapshot) {
        (self.0)(snapshot);
    }
}

impl fmt::Debug for LlmPayloadObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LlmPayloadObserver(..)")
    }
}

pub(crate) fn notify_llm_payload(
    options: &ChatOptions,
    round: u32,
    request_id: &str,
    messages: &[ChatMessage],
) {
    if let Some(observer) = &options.llm_payload_observer {
        observer.notify(LlmPayloadSnapshot {
            round,
            request_id: request_id.to_string(),
            model: options.model.clone(),
            system_prompt: options.system_prompt.clone(),
            messages: messages.to_vec(),
        });
    }
}

/// Provider and model settings for [`complete_with_tools`].
///
/// All fields beyond `base_url`, `api_key`, `model`, and `max_tool_rounds` are forwarded
/// directly to the OpenAI `POST /v1/chat/completions` body when set.
#[derive(Debug, Clone)]
pub struct ChatOptions {
    /// e.g. `https://api.openai.com` (no trailing slash required).
    pub base_url: String,
    /// API key. Stored as [`secrecy::SecretString`] — never appears in `Debug` output or logs.
    pub api_key: secrecy::SecretString,
    pub model: String,
    /// Maximum **HTTP completion** calls (each can include tool follow-up rounds).
    pub max_tool_rounds: u32,
    /// Prepended as a `"system"` message before all caller messages when set.
    pub system_prompt: Option<String>,

    // --- Sampling ---
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,

    // --- Token limits ---
    pub max_completion_tokens: Option<u32>,
    /// Maximum characters per tool-result message in chat history (`0` = no cap).
    pub tool_result_max_chars: usize,

    // --- Penalties ---
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,

    // --- Stop sequences ---
    pub stop: Option<StopSequence>,

    // --- Output format ---
    pub response_format: Option<ResponseFormat>,

    // --- Tool control ---
    pub tool_choice: Option<ToolChoice>,
    /// Forwarded to the chat Completions API when set. Omitting (`None`) leaves provider defaults.
    /// **Local** tool execution uses concurrent dispatch regardless — see the `join_all` path in [`complete_with_tools`].
    pub parallel_tool_calls: Option<bool>,

    // --- Logprobs ---
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,

    // --- Determinism / storage ---
    pub seed: Option<i64>,
    pub store: Option<bool>,

    // --- Service ---
    pub service_tier: Option<String>,

    // --- Reasoning models (o1/o3/o4) ---
    pub reasoning_effort: Option<String>,
    /// Responses API reasoning summary verbosity (`detailed` by default).
    pub reasoning_summary: reasoning::ReasoningSummaryLevel,

    /// JSON object merged into the chat completion request body after typed fields.
    pub extra_json: Option<Value>,

    /// Optional fan-out emitter for typed process events (`llm_call_*`, `tool_call_*`).
    pub status_emitter: Option<Arc<StatusEmitter>>,

    /// Optional observer invoked before each LLM HTTP call with the full message list.
    pub llm_payload_observer: Option<LlmPayloadObserver>,

    // --- Correlation ---
    /// Caller-supplied request identifier used for tracing and correlation.
    /// Auto-generated as a UUID v4 if `None` when the request is executed.
    pub request_id: Option<String>,

    // --- Cooperative cancellation ---
    /// When set, all HTTP calls in this request check the token and return
    /// [`ChatError::Cancelled`] immediately if it has been cancelled.
    pub cancel: Option<CancellationToken>,

    /// Optional ordered model list (primary first). On eligible HTTP failures after retries,
    /// the next model is attempted.
    pub model_fallback: Option<crate::fallback::ModelFallbackChain>,

    /// Multi-provider API keys (when set, used instead of legacy `api_key` / `base_url` alone).
    pub provider_credentials: Option<Arc<crate::providers::ProviderCredentials>>,

    // --- Context optimization (GlueLLM parity) ---
    pub tool_mode: ToolMode,
    pub tool_route_model: Option<String>,
    pub condense_tool_messages: bool,
    pub aaak_tool_condensing: bool,
    pub summarize_context: SummarizeContextConfig,
    pub aaak_compression_enabled: bool,
    pub aaak_compression_model: Option<String>,
    /// Optional compact context block appended after each condensed tool round.
    pub context_block_provider: Option<ContextBlockProvider>,
    /// Optional sink for context lifecycle diagnostics.
    pub context_event_logger: Option<ContextEventLogger>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions {
            base_url: String::new(),
            api_key: secrecy::SecretString::from(String::new()),
            model: String::new(),
            max_tool_rounds: 0,
            system_prompt: None,
            temperature: None,
            top_p: None,
            n: None,
            max_completion_tokens: None,
            tool_result_max_chars: 32_000,
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            response_format: None,
            tool_choice: None,
            parallel_tool_calls: None,
            logprobs: None,
            top_logprobs: None,
            seed: None,
            store: None,
            service_tier: None,
            reasoning_effort: None,
            reasoning_summary: reasoning::ReasoningSummaryLevel::default(),
            extra_json: None,
            status_emitter: None,
            llm_payload_observer: None,
            request_id: None,
            cancel: None,
            model_fallback: None,
            provider_credentials: None,
            tool_mode: ToolMode::default(),
            tool_route_model: None,
            condense_tool_messages: false,
            aaak_tool_condensing: false,
            summarize_context: SummarizeContextConfig::default(),
            aaak_compression_enabled: false,
            aaak_compression_model: None,
            context_block_provider: None,
            context_event_logger: None,
        }
    }
}

impl ChatOptions {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        ChatOptions {
            base_url: base_url.into(),
            api_key: secrecy::SecretString::from(api_key.into()),
            model: model.into(),
            max_tool_rounds: 16,
            ..Default::default()
        }
    }
}

impl From<proto::ChatOptions> for ChatOptions {
    fn from(p: proto::ChatOptions) -> Self {
        ChatOptions {
            base_url: if p.base_url.is_empty() {
                "https://api.openai.com".to_string()
            } else {
                p.base_url
            },
            api_key: secrecy::SecretString::from(p.api_key),
            model: if p.model.is_empty() {
                "gpt-5.4-nano-2026-03-17-mini".to_string()
            } else {
                p.model
            },
            max_tool_rounds: if p.max_tool_rounds == 0 {
                16
            } else {
                p.max_tool_rounds
            },
            system_prompt: p.system_prompt,
            temperature: p.temperature,
            top_p: p.top_p,
            n: None,
            max_completion_tokens: p.max_completion_tokens,
            tool_result_max_chars: 32_000,
            presence_penalty: p.presence_penalty,
            frequency_penalty: p.frequency_penalty,
            stop: p.stop.map(StopSequence::One),
            response_format: None,
            tool_choice: None,
            parallel_tool_calls: p.parallel_tool_calls,
            logprobs: p.logprobs,
            top_logprobs: p.top_logprobs,
            seed: p.seed,
            store: p.store,
            service_tier: p.service_tier,
            reasoning_effort: p.reasoning_effort,
            reasoning_summary: reasoning::ReasoningSummaryLevel::default(),
            extra_json: p
                .extra_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
            status_emitter: None,
            llm_payload_observer: None,
            request_id: None,
            cancel: None,
            model_fallback: None,
            provider_credentials: None,
            tool_mode: parse_tool_mode(p.tool_mode.as_deref()),
            tool_route_model: p.tool_route_model,
            condense_tool_messages: p.condense_tool_messages.unwrap_or(false),
            aaak_tool_condensing: p.aaak_tool_condensing.unwrap_or(false),
            summarize_context: SummarizeContextConfig {
                enabled: p.summarize_context_enabled.unwrap_or(false),
                threshold: usize::try_from(p.summarize_context_threshold.unwrap_or(20))
                    .unwrap_or(20),
                keep_recent: usize::try_from(p.summarize_context_keep_recent.unwrap_or(12))
                    .unwrap_or(12),
                max_chars: 800_000,
            },
            aaak_compression_enabled: p.aaak_compression_enabled.unwrap_or(false),
            aaak_compression_model: p.aaak_compression_model,
            context_block_provider: None,
            context_event_logger: None,
        }
    }
}

fn parse_tool_mode(s: Option<&str>) -> ToolMode {
    match s {
        Some("dynamic") => ToolMode::Dynamic,
        _ => ToolMode::Standard,
    }
}

/// Outcome of a completed turn (assistant returned text or empty after tool loop).
#[derive(Debug, Clone)]
pub struct CompletionOutcome {
    pub content: Option<String>,
    /// Number of completion HTTP calls performed.
    pub rounds: u32,
    /// Token usage reported by the final completion response.
    pub usage: Option<proto::Usage>,
    /// The `finish_reason` from the last choice.
    pub finish_reason: Option<String>,
    /// Correlation ID for this request (UUID v4 auto-generated if not supplied by caller).
    pub request_id: String,
    /// Full caller-visible chat history after this turn (user / assistant / tool roles).
    ///
    /// Excludes the synthetic leading `system` row derived from [`ChatOptions::system_prompt`]
    /// when that option is set, so this slice can be passed back into [`complete_with_tools`]
    /// on the next turn.
    pub messages: Vec<ChatMessage>,
    /// Model that produced the final successful HTTP response (after any fallback).
    pub model_used: String,
}

/// Strips the leading `system` message when it was injected from [`ChatOptions::system_prompt`].
#[must_use]
pub fn conversation_messages_for_client(
    messages: &[ChatMessage],
    had_system_prompt: bool,
) -> Vec<ChatMessage> {
    if had_system_prompt && messages.first().is_some_and(|m| m.role == "system") {
        messages[1..].to_vec()
    } else {
        messages.to_vec()
    }
}

fn terminal_assistant_for_history(
    model_msg: &ChatMessage,
    final_content: &Option<String>,
) -> ChatMessage {
    let model_text = model_msg.content.as_ref().and_then(MessageContent::as_text);
    match final_content {
        Some(t) if model_text != Some(t.as_str()) => {
            let mut m = model_msg.clone();
            m.content = Some(MessageContent::Text(t.clone()));
            m
        }
        _ => model_msg.clone(),
    }
}

/// Errors from the chat + tool pipeline.
#[derive(Debug, Error)]
pub enum ChatError {
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Tool(#[from] ToolInvokeError),
    #[error("hook aborted: {0}")]
    Hook(#[from] HookError),
    #[error(transparent)]
    Guardrail(#[from] GuardrailError),
    #[error("completion response contained no choices")]
    NoChoice,
    /// Provider finished a round with neither assistant text nor tool calls.
    /// Common intermittent failure on weaker models / some Groq tool-calling streams.
    #[error("model returned an empty response (no text, no tool calls)")]
    EmptyResponse,
    #[error("exceeded max tool rounds ({0})")]
    MaxToolRounds(u32),
    #[error("request cancelled")]
    Cancelled,
    #[error(transparent)]
    Credentials(#[from] crate::providers::CredentialsError),
    #[error("unsupported provider for this API: {0}")]
    UnsupportedProvider(crate::providers::ProviderId),
    #[error("API response failed: {0}")]
    Api(String),
    /// Tool loop ended before a successful completion; carries transcript for persistence.
    #[error("{cause}")]
    PartialTurn {
        #[source]
        cause: Box<ChatError>,
        messages: Vec<ChatMessage>,
    },
}

impl ChatError {
    /// Client-facing messages accumulated before the failure, if any.
    pub fn partial_messages(&self) -> Option<&[ChatMessage]> {
        match self {
            ChatError::PartialTurn { messages, .. } => Some(messages),
            _ => None,
        }
    }

    /// Inner error when wrapped in [`PartialTurn`]; otherwise `self`.
    pub fn root_cause(&self) -> &ChatError {
        match self {
            ChatError::PartialTurn { cause, .. } => cause.as_ref(),
            other => other,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self.root_cause(), ChatError::Cancelled)
    }
}

/// Wrap a tool-loop failure with the messages accumulated so far.
pub fn fail_partial(
    cause: ChatError,
    messages: &[ChatMessage],
    had_system_prompt: bool,
) -> ChatError {
    ChatError::PartialTurn {
        cause: Box::new(cause),
        messages: conversation_messages_for_client(messages, had_system_prompt),
    }
}

/// Resolve credentials from options (multi-provider map or legacy single OpenAI key).
pub fn credentials_for(options: &ChatOptions) -> crate::providers::ProviderCredentials {
    if let Some(creds) = &options.provider_credentials {
        return creds.as_ref().clone();
    }
    let mut creds = crate::providers::ProviderCredentials::new();
    creds.with_legacy_openai_key(options.api_key.expose_secret(), Some(&options.base_url));
    creds
}

/// Perform one HTTP JSON POST, honouring an optional cancellation token.
async fn post_json_cancellable(
    http: &HttpClient,
    url: &str,
    body: &Value,
    headers: &[(&str, &str)],
    rate_limit_key: Option<crate::providers::RateLimitKey>,
    cancel: Option<&CancellationToken>,
) -> Result<Value, ChatError> {
    if let Some(token) = cancel {
        tokio::select! {
            biased;
            _ = token.cancelled() => Err(ChatError::Cancelled),
            result = http.post_json_with_headers(url, body, headers, rate_limit_key) => Ok(result?),
        }
    } else {
        Ok(http
            .post_json_with_headers(url, body, headers, rate_limit_key)
            .await?)
    }
}

/// POST via provider adapter with per-model HTTP retries and optional model fallback chain.
pub(crate) async fn provider_chat_post(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[crate::tools::ToolSpec]>,
    chat_tools: Option<&[crate::openai::ChatTool]>,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
    stream: bool,
) -> Result<(Value, crate::providers::ModelRef), ChatError> {
    let models = crate::fallback::effective_models(&options.model, options.model_fallback.as_ref());
    let default_policy = crate::fallback::FallbackPolicy::default();
    let policy = options
        .model_fallback
        .as_ref()
        .map(|c| &c.policy)
        .unwrap_or(&default_policy);

    let mut last_err = None;
    for (i, model_str) in models.iter().enumerate() {
        if i > 0 {
            crate::fallback::emit_model_fallback(
                options.status_emitter.as_ref(),
                request_id,
                &models[i - 1],
                model_str,
                round,
            )
            .await;
        }

        let model_ref = crate::providers::parse_model_ref(model_str);
        credentials
            .key_for(model_ref.provider)
            .map_err(ChatError::Credentials)?;

        let provider = crate::providers::resolve_provider(&model_ref);
        let ctx = crate::providers::ProviderRequestContext {
            model_ref: &model_ref,
            credentials,
            messages,
            tools: tool_specs,
            chat_tools,
            stream,
            options,
        };
        let req = provider.build_chat_request(&ctx);
        let header_refs: Vec<(&str, &str)> = req
            .headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        match post_json_cancellable(
            http,
            &req.url,
            &req.body,
            &header_refs,
            Some(req.rate_limit_key),
            options.cancel.as_ref(),
        )
        .await
        {
            Ok(v) => return Ok((v, model_ref)),
            Err(e) => {
                if i + 1 < models.len()
                    && crate::fallback::chat_error_eligible_for_fallback(&e, policy)
                {
                    last_err = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }
    Err(last_err.unwrap_or(ChatError::Http(HttpError::InvalidJson(
        "model fallback exhausted".into(),
    ))))
}

/// Open a streaming POST via provider adapter with per-model HTTP retries and optional fallback.
pub(crate) async fn provider_chat_stream(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[crate::tools::ToolSpec]>,
    chat_tools: Option<&[crate::openai::ChatTool]>,
    options: &ChatOptions,
    request_id: &str,
    round: u32,
) -> Result<
    (
        impl futures_util::Stream<Item = Result<bytes::Bytes, HttpError>> + Send + use<>,
        crate::providers::ModelRef,
    ),
    ChatError,
> {
    let models = crate::fallback::effective_models(&options.model, options.model_fallback.as_ref());
    let default_policy = crate::fallback::FallbackPolicy::default();
    let policy = options
        .model_fallback
        .as_ref()
        .map(|c| &c.policy)
        .unwrap_or(&default_policy);

    let mut last_err = None;
    for (i, model_str) in models.iter().enumerate() {
        if i > 0 {
            crate::fallback::emit_model_fallback(
                options.status_emitter.as_ref(),
                request_id,
                &models[i - 1],
                model_str,
                round,
            )
            .await;
        }

        let model_ref = crate::providers::parse_model_ref(model_str);
        credentials
            .key_for(model_ref.provider)
            .map_err(ChatError::Credentials)?;

        let provider = crate::providers::resolve_provider(&model_ref);
        let ctx = crate::providers::ProviderRequestContext {
            model_ref: &model_ref,
            credentials,
            messages,
            tools: tool_specs,
            chat_tools,
            stream: true,
            options,
        };
        let req = provider.build_chat_request(&ctx);
        let url = req.url.clone();
        let body = req.body.clone();
        let owned_headers = req.headers.clone();
        let rate_key = req.rate_limit_key;
        let header_refs: Vec<(&str, &str)> = owned_headers
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        if let Some(token) = options.cancel.as_ref()
            && token.is_cancelled()
        {
            return Err(ChatError::Cancelled);
        }

        match http
            .post_json_stream_with_headers(&url, &body, &header_refs, Some(rate_key))
            .await
        {
            Ok(stream) => return Ok((stream, model_ref)),
            Err(e) => {
                let chat_err = ChatError::Http(e);
                if i + 1 < models.len()
                    && crate::fallback::chat_error_eligible_for_fallback(&chat_err, policy)
                {
                    last_err = Some(chat_err);
                    continue;
                }
                return Err(chat_err);
            }
        }
    }
    Err(last_err.unwrap_or(ChatError::Http(HttpError::InvalidJson(
        "model fallback exhausted".into(),
    ))))
}

/// Gateway-facing wrapper around [`provider_chat_post`].
#[cfg(feature = "gateway")]
pub async fn proxy_chat_post(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[crate::tools::ToolSpec]>,
    options: &ChatOptions,
    request_id: &str,
) -> Result<(Value, crate::providers::ModelRef), ChatError> {
    provider_chat_post(
        http,
        credentials,
        messages,
        tool_specs,
        None,
        options,
        request_id,
        1,
        false,
    )
    .await
}

/// Gateway-facing wrapper around [`provider_chat_stream`].
#[cfg(feature = "gateway")]
pub async fn proxy_chat_stream(
    http: &HttpClient,
    credentials: &crate::providers::ProviderCredentials,
    messages: &[ChatMessage],
    tool_specs: Option<&[crate::tools::ToolSpec]>,
    options: &ChatOptions,
    request_id: &str,
) -> Result<
    (
        impl futures_util::Stream<Item = Result<bytes::Bytes, HttpError>> + Send + use<>,
        crate::providers::ModelRef,
    ),
    ChatError,
> {
    provider_chat_stream(
        http,
        credentials,
        messages,
        tool_specs,
        None,
        options,
        request_id,
        1,
    )
    .await
}

/// POST JSON with per-model HTTP retries and optional model fallback chain.
pub(crate) async fn post_json_with_model_fallback(
    http: &HttpClient,
    url: &str,
    body: &Value,
    headers: &[(&str, &str)],
    options: &ChatOptions,
    request_id: &str,
    round: u32,
    rate_limit_key: Option<crate::providers::RateLimitKey>,
) -> Result<(Value, String), ChatError> {
    let models = crate::fallback::effective_models(&options.model, options.model_fallback.as_ref());
    let default_policy = crate::fallback::FallbackPolicy::default();
    let policy = options
        .model_fallback
        .as_ref()
        .map(|c| &c.policy)
        .unwrap_or(&default_policy);

    let mut last_err = None;
    for (i, model) in models.iter().enumerate() {
        if i > 0 {
            crate::fallback::emit_model_fallback(
                options.status_emitter.as_ref(),
                request_id,
                &models[i - 1],
                model,
                round,
            )
            .await;
        }
        let mut body = body.clone();
        crate::fallback::set_body_model(&mut body, model);

        match post_json_cancellable(
            http,
            url,
            &body,
            headers,
            rate_limit_key,
            options.cancel.as_ref(),
        )
        .await
        {
            Ok(v) => return Ok((v, model.clone())),
            Err(e) => {
                if i + 1 < models.len()
                    && crate::fallback::chat_error_eligible_for_fallback(&e, policy)
                {
                    last_err = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }
    Err(last_err.unwrap_or(ChatError::Http(HttpError::InvalidJson(
        "model fallback exhausted".into(),
    ))))
}

pub(crate) fn observation_hook_ctx(
    stage: HookStage,
    content: String,
    request_id: &str,
    round: u32,
    model: &str,
) -> HookContext {
    let mut ctx = HookContext::new(stage, content);
    ctx.metadata.insert(
        "request_id".to_string(),
        Value::String(request_id.to_string()),
    );
    ctx.metadata
        .insert("round".to_string(), Value::Number(round.into()));
    ctx.metadata
        .insert("model".to_string(), Value::String(model.to_string()));
    ctx
}

fn http_error_type(err: &ChatError) -> String {
    match err.root_cause() {
        ChatError::Http(e) => format!("http:{e}"),
        ChatError::Cancelled => "cancelled".to_string(),
        ChatError::Serde(e) => format!("serde:{e}"),
        ChatError::NoChoice => "no_choice".to_string(),
        other => format!("{other}"),
    }
}

/// Invoke a tool, retrying on [`ToolInvokeError::HandlerFailed`] according to `policy`.
async fn invoke_with_policy(
    tool: &std::sync::Arc<dyn crate::tools::Tool>,
    name: &str,
    arguments: Value,
    policy: &ToolRetryPolicy,
) -> Result<Value, ToolInvokeError> {
    let result = tool.call(arguments.clone()).await;

    match result {
        Ok(v) => {
            metrics::counter!(crate::telemetry::metrics::TOOL_CALLS_TOTAL, "tool_name" => name.to_string()).increment(1);
            Ok(v)
        }
        Err(ToolInvokeError::HandlerFailed { ref message, .. }) => {
            metrics::counter!(crate::telemetry::metrics::TOOL_CALLS_ERRORS, "tool_name" => name.to_string(), "error_kind" => "handler_failed").increment(1);
            match &policy.on_error {
                OnToolError::FailFast => Err(result.unwrap_err()),
                OnToolError::Skip => {
                    tracing::warn!(tool = name, error = %message, "tool failed (skip policy)");
                    Ok(json!({
                        "ok": false,
                        "error": message,
                    }))
                }
                OnToolError::Retry {
                    max,
                    initial_delay_ms,
                } => {
                    let mut delay = *initial_delay_ms;
                    for attempt in 1..=*max {
                        tracing::warn!(
                            tool = name,
                            attempt,
                            max,
                            delay_ms = delay,
                            "tool failed, retrying"
                        );
                        sleep(Duration::from_millis(delay)).await;
                        delay = delay.saturating_mul(2);

                        match tool.call(arguments.clone()).await {
                            Ok(v) => {
                                metrics::counter!(crate::telemetry::metrics::TOOL_CALLS_TOTAL, "tool_name" => name.to_string()).increment(1);
                                return Ok(v);
                            }
                            Err(ToolInvokeError::HandlerFailed { .. }) if attempt < *max => {
                                continue;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    Err(result.unwrap_err())
                }
            }
        }
        Err(other) => {
            metrics::counter!(crate::telemetry::metrics::TOOL_CALLS_ERRORS, "tool_name" => name.to_string(), "error_kind" => "other").increment(1);
            Err(other)
        }
    }
}

/// Cap tool result/argument payload size in process-event metadata (UI / observability).
const TOOL_EVENT_METADATA_MAX_BYTES: usize = 16_384;

fn truncate_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &value[..end])
}

pub(crate) fn truncate_tool_event_metadata(value: &str) -> String {
    truncate_bytes(value, TOOL_EVENT_METADATA_MAX_BYTES)
}

/// Truncate tool-result text before it is appended to chat history.
pub(crate) fn truncate_tool_result(content: String, max_chars: usize) -> String {
    if max_chars == 0 || content.len() <= max_chars {
        return content;
    }
    format!("{}…\n[truncated]", truncate_bytes(&content, max_chars))
}

/// Dispatch a single `"function"` tool call through the full pre/post hook pipeline.
///
/// Returns the [`ChatMessage`] with `role="tool"` that should be appended to the
/// conversation history. Called concurrently for all tool calls in a single round
/// via [`futures_util::future::join_all`].
#[instrument(
    skip(tc, hooks, registry, status_emitter),
    fields(tool.name = %tc.function.name, tool.id = %tc.id)
)]
pub(crate) async fn dispatch_one(
    tc: &ToolCall,
    hooks: &HookRegistry,
    registry: &ToolRegistry,
    status_emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    round: u32,
    model: &str,
    emit_start: bool,
    tool_result_max_chars: usize,
    loop_guard: Option<&SharedToolLoopGuard>,
) -> Result<ChatMessage, ChatError> {
    let tool_name = tc.function.name.clone();
    if emit_start && let Some(emitter) = status_emitter {
        let mut ev = ProcessEvent::new(ProcessEventKind::ToolCallStart, request_id, model);
        ev.round = round;
        ev.metadata
            .insert("tool_name".to_string(), tool_name.clone());
        ev.metadata.insert(
            "arguments".to_string(),
            truncate_tool_event_metadata(tc.function.arguments.trim()),
        );
        emit_safe(Some(emitter), ev).await;
    }

    let exec_result: Result<String, ChatError> = async {
        let pre_ctx = if hooks.is_empty_for(&HookStage::PreTool).await {
            HookContext::with_meta(
                HookStage::PreTool,
                tc.function.arguments.trim(),
                "tool_name",
                tc.function.name.as_str(),
            )
        } else {
            hooks
                .run(
                    HookStage::PreTool,
                    HookContext::with_meta(
                        HookStage::PreTool,
                        tc.function.arguments.trim(),
                        "tool_name",
                        tc.function.name.as_str(),
                    ),
                )
                .await?
        };
        if pre_ctx.metadata.get("tool_skip").and_then(|v| v.as_bool()) == Some(true) {
            let skip_result = pre_ctx
                .metadata
                .get("tool_skip_result")
                .and_then(|v| v.as_str())
                .unwrap_or("{\"ok\":false,\"error\":\"tool call blocked by hook\"}");
            return Ok(skip_result.to_string());
        }
        // If the LLM's argument JSON was truncated (e.g. by token limits), return a
        // soft error as a tool result so the model can retry rather than killing the turn.
        // An empty arguments string is treated as `{}` because some models omit braces for
        // no-parameter tools.
        let args: Value = {
            let raw = pre_ctx.content.trim();
            if raw.is_empty() {
                Value::Object(Default::default())
            } else {
                match serde_json::from_str(raw) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(
                            tool = %tc.function.name,
                            error = %e,
                            "malformed tool-call JSON — returning error to model"
                        );
                        return Ok(serde_json::to_string(&serde_json::json!({
                            "ok": false,
                            "error": format!("tool call arguments were not valid JSON ({e}). \
                                Please retry with complete, well-formed JSON arguments.")
                        }))?);
                    }
                }
            }
        };
        let (tool, policy) = registry
            .resolve_invocation(&tc.function.name)
            .await
            .map_err(ChatError::Tool)?;

        if let Some(guard) = loop_guard {
            let mut g = guard.lock().await;
            if let Some(cached) = g.check_cached(&tc.function.name, &args) {
                return Ok(cached);
            }
        }

        // Serialize same-file mutations so concurrent edits in one round cannot
        // invalidate each other's anchors.
        let _file_guard = file_locks::acquire_file_lock(&tc.function.name, &args).await;

        let result = invoke_with_policy(&tool, &tc.function.name, args.clone(), &policy).await?;
        let result_json = serde_json::to_string(&result)?;

        if let Some(guard) = loop_guard {
            guard
                .lock()
                .await
                .record(&tc.function.name, &args, &result_json);
        }

        let mut post_ctx = HookContext::with_meta(
            HookStage::PostTool,
            &result_json,
            "tool_name",
            tc.function.name.as_str(),
        );
        post_ctx.metadata.insert(
            "context_policy".into(),
            json!(tool.context_policy().as_str()),
        );
        post_ctx
            .metadata
            .insert("arguments".into(), json!(tc.function.arguments.trim()));
        let post_ctx = if hooks.is_empty_for(&HookStage::PostTool).await {
            post_ctx
        } else {
            hooks.run(HookStage::PostTool, post_ctx).await?
        };
        let offload_ref = serde_json::from_str::<Value>(&post_ctx.content)
            .ok()
            .and_then(|value| {
                value
                    .get("notepad_ref")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| {
                        value
                            .get("notepad_refs")
                            .and_then(Value::as_array)
                            .and_then(|refs| refs.first())
                            .and_then(|reference| reference.get("notepad_ref"))
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
            });
        if post_ctx.content != result_json && offload_ref.is_some() {
            tracing::info!(
                tool = %tc.function.name,
                source_chars = result_json.chars().count(),
                stub_chars = post_ctx.content.chars().count(),
                entry_id = %offload_ref.as_deref().unwrap_or(""),
                "tool result offloaded to notepad"
            );
        }
        Ok(post_ctx.content)
    }
    .await;

    if let Some(emitter) = status_emitter {
        let mut ev = ProcessEvent::new(ProcessEventKind::ToolCallEnd, request_id, model);
        ev.round = round;
        ev.metadata
            .insert("tool_name".to_string(), tool_name.clone());
        ev.metadata.insert(
            "arguments".to_string(),
            truncate_tool_event_metadata(tc.function.arguments.trim()),
        );
        match &exec_result {
            Ok(content) => {
                if let Some(summary) = tool_summary::tool_result_summary(&tool_name, content) {
                    ev.metadata.insert("result_summary".to_string(), summary);
                }
                ev.metadata
                    .insert("result".to_string(), truncate_tool_event_metadata(content));
            }
            Err(err) => {
                ev.error_type = Some(err.to_string());
                ev.metadata.insert(
                    "result".to_string(),
                    truncate_tool_event_metadata(&err.to_string()),
                );
            }
        }
        emit_safe(Some(emitter), ev).await;
    }

    let content = truncate_tool_result(exec_result?, tool_result_max_chars);
    Ok(ChatMessage {
        role: "tool".to_string(),
        content: Some(MessageContent::Text(content)),
        tool_calls: None,
        tool_call_id: Some(tc.id.clone()),
        name: Some(tool_name),
        refusal: None,
    })
}

// ---------------------------------------------------------------------------
// complete_with_tools
// ---------------------------------------------------------------------------

/// Run OpenAI-style chat completions with tools: calls `POST /v1/chat/completions` until the model
/// returns an assistant message without tool calls or [`ChatOptions::max_tool_rounds`] is exceeded.
///
/// **Input guardrails** run on the last user message before the first LLM call.
/// **Output guardrails** run on the final assistant text with a retry loop
/// (up to [`GuardrailRegistry::max_output_retries`]).
#[instrument(
    skip(http, registry, hooks, guardrails, caller_messages, options),
    fields(model = %options.model)
)]
pub async fn complete_with_tools(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    caller_messages: Vec<ChatMessage>,
    options: &ChatOptions,
) -> Result<CompletionOutcome, ChatError> {
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    tracing::info!(
        request_id = %request_id,
        model = %options.model,
        "complete_with_tools started"
    );
    metrics::counter!(crate::telemetry::metrics::COMPLETIONS_TOTAL, "model" => options.model.clone()).increment(1);

    let credentials = credentials_for(options);

    // Prepend system prompt if configured.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        messages.push(ChatMessage::text("system", sp));
    }
    messages.extend(caller_messages);
    let had_system_prompt = options.system_prompt.is_some();

    // --- Input guardrails: run on the last user message before first LLM call ---
    if !guardrails.input_is_empty().await {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();

        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user {
                    if let Some(msg) = messages.iter_mut().rev().find(|m| m.role == "user") {
                        msg.content = Some(MessageContent::Text(transformed));
                    }
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(fail_partial(
                    ChatError::Guardrail(GuardrailError::new(
                        GuardrailStage::Input,
                        guard_name,
                        reason,
                    )),
                    &messages,
                    had_system_prompt,
                ));
            }
        }
    }

    let mut api_calls: u32 = 0;
    let mut model_used;
    let mut accumulated_usage: Option<proto::Usage> = None;
    // One automatic re-request when a provider returns a blank round (Groq flake).
    let mut empty_round_retries: u32 = 0;
    let loop_guard = new_tool_loop_guard();

    let all_specs = registry.list_specs().await;
    let mut active_set = ActiveToolSet::new(all_specs, options.tool_mode);
    let route_model = options
        .tool_route_model
        .as_deref()
        .unwrap_or(DEFAULT_TOOL_ROUTE_MODEL);

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            metrics::counter!(crate::telemetry::metrics::COMPLETIONS_ERRORS, "model" => options.model.clone(), "error_kind" => "max_tool_rounds").increment(1);
            return Err(fail_partial(
                ChatError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                had_system_prompt,
            ));
        }
        api_calls += 1;
        if let Some(logger) = &options.context_event_logger {
            logger.log(format!(
                "context round={} messages={} chars={}",
                api_calls,
                messages.len(),
                estimate_context_chars(&messages),
            ));
        }
        tracing::info!(
            round = api_calls,
            request_id = %request_id,
            "llm_completion_round"
        );

        // --- PreCompletion hook (observation only) ---
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        hooks
            .run(
                HookStage::PreCompletion,
                observation_hook_ctx(
                    HookStage::PreCompletion,
                    last_user.clone(),
                    &request_id,
                    api_calls,
                    &options.model,
                ),
            )
            .await
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, had_system_prompt))?;

        maybe_summarize_messages(
            http,
            &credentials,
            &mut messages,
            &options.summarize_context,
            options.aaak_compression_enabled,
            options.aaak_compression_model.as_deref(),
            &options.model,
            options,
            &request_id,
        )
        .await
        .map_err(|e| fail_partial(e, &messages, had_system_prompt))?;

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        notify_llm_payload(options, api_calls, &request_id, &messages);

        let tool_specs = active_set.specs_for_llm();
        let chat_tools = active_set.chat_tools_for_llm();

        let (val, model_ref) = match provider_chat_post(
            http,
            &credentials,
            &messages,
            tool_specs,
            chat_tools,
            options,
            &request_id,
            api_calls,
            false,
        )
        .await
        {
            Ok(pair) => pair,
            Err(e) => {
                let mut ev =
                    ProcessEvent::new(ProcessEventKind::LlmCallError, &request_id, &options.model);
                ev.round = api_calls;
                ev.error_type = Some(http_error_type(&e));
                emit_safe(options.status_emitter.as_ref(), ev).await;
                return Err(fail_partial(e, &messages, had_system_prompt));
            }
        };
        model_used = model_ref.raw.clone();
        let provider = crate::providers::resolve_provider(&model_ref);
        let normalized = provider.parse_chat_response(&val).map_err(|e| {
            fail_partial(if e.to_string().contains("no choices") {
                ChatError::NoChoice
            } else {
                ChatError::Http(HttpError::InvalidJson(e.to_string()))
            }, &messages, had_system_prompt)
        })?;

        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: normalized.content.clone().map(MessageContent::Text),
            tool_calls: if normalized.tool_calls.is_empty() {
                None
            } else {
                Some(normalized.tool_calls.clone())
            },
            tool_call_id: None,
            name: None,
            refusal: None,
        };

        let tool_call_count = normalized
            .tool_calls
            .iter()
            .filter(|tc| tc.kind == "function")
            .count() as u32;
        let usage_proto = normalized.usage.clone();
        if let Some(ref u) = usage_proto {
            accumulated_usage = Some(crate::usage::accumulate_usage(
                accumulated_usage.as_ref(),
                u,
            ));
        }
        let estimated_cost = usage_proto
            .as_ref()
            .map(|u| estimate_model_call_cost_usd(&model_used, u));

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
            ev.round = api_calls;
            ev.tool_call_count = tool_call_count;
            ev.usage = usage_proto.clone();
            ev.estimated_cost_usd = estimated_cost;
            ev
        })
        .await;

        // --- PostCompletion hook (observation only) ---
        let assistant_text = normalized.content.clone().unwrap_or_default();
        hooks
            .run(
                HookStage::PostCompletion,
                observation_hook_ctx(
                    HookStage::PostCompletion,
                    assistant_text,
                    &request_id,
                    api_calls,
                    &model_used,
                ),
            )
            .await
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, had_system_prompt))?;

        if !normalized.tool_calls.is_empty() {
            let function_tcs: Vec<_> = normalized
                .tool_calls
                .iter()
                .filter(|tc| tc.kind == "function")
                .collect();

            if active_set.has_router() && is_router_call(&normalized.tool_calls) {
                let query = router_query_from_calls(&normalized.tool_calls)
                    .unwrap_or_else(|| last_user.clone());
                let user_context = user_context_for_route(&messages, &query);
                let matched = resolve_tool_route(
                    http,
                    &credentials,
                    &user_context,
                    active_set.dynamic_specs(),
                    route_model,
                    options,
                    &request_id,
                )
                .await;
                active_set.apply_route(matched);
                let matched_names: Vec<String> = active_set
                    .specs_for_llm()
                    .unwrap_or(&[])
                    .iter()
                    .filter(|s| !s.static_tool && s.name != crate::tools::ROUTER_TOOL_NAME)
                    .map(|s| s.name.clone())
                    .collect();
                emit_safe(options.status_emitter.as_ref(), {
                    let mut ev =
                        ProcessEvent::new(ProcessEventKind::ToolRoute, &request_id, route_model);
                    ev.round = api_calls;
                    ev.metadata.insert("route_query".to_string(), query.clone());
                    ev.metadata
                        .insert("matched_tools".to_string(), matched_names.join(","));
                    ev
                })
                .await;
                continue;
            }

            messages.push(msg.clone());

            // Run tool handlers concurrently on this process (order of `tool` messages follows `tool_calls`).
            tracing::info!(
                count = function_tcs.len(),
                request_id = %request_id,
                "tool_calls_batch"
            );
            let results = futures_util::future::join_all(function_tcs.iter().map(|tc| {
                dispatch_one(
                    tc,
                    hooks,
                    registry,
                    options.status_emitter.as_ref(),
                    &request_id,
                    api_calls,
                    &options.model,
                    true,
                    options.tool_result_max_chars,
                    Some(&loop_guard),
                )
            }))
            .await;
            for r in results {
                messages.push(r.map_err(|e| fail_partial(e, &messages, had_system_prompt))?);
            }

            if options.condense_tool_messages {
                condense_tool_round(
                    &mut messages,
                    options.aaak_tool_condensing,
                    options.context_block_provider.as_ref(),
                    options.context_event_logger.as_ref(),
                );
            }
            continue;
        }

        // --- Terminal response: extract content ---
        let content_empty = normalized
            .content
            .as_ref()
            .is_none_or(|s| s.trim().is_empty());
        if content_empty {
            if empty_round_retries < 1 {
                empty_round_retries += 1;
                // Don't burn a max_tool_rounds slot on a blank provider response.
                api_calls = api_calls.saturating_sub(1);
                tracing::warn!(
                    request_id = %request_id,
                    model = %model_used,
                    finish_reason = ?normalized.finish_reason,
                    "empty LLM round (no content, no tool calls) — retrying once"
                );
                continue;
            }
            tracing::warn!(
                request_id = %request_id,
                model = %model_used,
                finish_reason = ?normalized.finish_reason,
                "empty LLM round after retry — failing turn"
            );
            return Err(fail_partial(
                ChatError::EmptyResponse,
                &messages,
                had_system_prompt,
            ));
        }

        let usage = accumulated_usage.clone();
        let content = normalized.content.clone();
        let finish_reason = normalized.finish_reason.clone();

        // --- Output guardrails: retry loop ---
        if guardrails.output_is_empty().await {
            messages.push(msg.clone());
            let client_messages = conversation_messages_for_client(&messages, had_system_prompt);
            break CompletionOutcome {
                content,
                rounds: api_calls,
                usage,
                finish_reason,
                request_id: request_id.clone(),
                messages: client_messages,
                model_used: model_used.clone(),
            };
        }

        let text = content.clone().unwrap_or_default();
        let saved_content = content;
        let saved_finish = finish_reason;
        let max_retries = guardrails.max_output_retries;

        let mut final_outcome = None;
        for attempt in 0..=max_retries {
            let (outcome, guard_name) = guardrails.run_output(&text).await;
            match outcome {
                GuardrailOutcome::Allow(transformed) => {
                    let final_content = if transformed == text {
                        saved_content.clone()
                    } else {
                        Some(transformed)
                    };
                    messages.push(terminal_assistant_for_history(&msg, &final_content));
                    let client_messages =
                        conversation_messages_for_client(&messages, had_system_prompt);
                    final_outcome = Some(CompletionOutcome {
                        content: final_content,
                        rounds: api_calls,
                        usage,
                        finish_reason: saved_finish.clone(),
                        request_id: request_id.clone(),
                        messages: client_messages,
                        model_used: model_used.clone(),
                    });
                    break;
                }
                GuardrailOutcome::Block(reason) => {
                    if attempt < max_retries {
                        messages.push(msg.clone());
                        messages.push(ChatMessage::text(
                            "user",
                            format!(
                                "Your previous response was rejected by a content policy ({reason}). \
                                 Please revise it."
                            ),
                        ));
                        if api_calls >= options.max_tool_rounds {
                            metrics::counter!(crate::telemetry::metrics::COMPLETIONS_ERRORS, "model" => options.model.clone(), "error_kind" => "guardrail").increment(1);
                            return Err(fail_partial(
                                ChatError::Guardrail(GuardrailError::new(
                                    GuardrailStage::Output,
                                    guard_name,
                                    reason,
                                )),
                                &messages,
                                had_system_prompt,
                            ));
                        }
                        api_calls += 1;
                        let credentials = credentials_for(options);
                        let (val, model_ref) = provider_chat_post(
                            http,
                            &credentials,
                            &messages,
                            None,
                            None,
                            options,
                            &request_id,
                            api_calls,
                            false,
                        )
                        .await
                        .map_err(|e| fail_partial(e, &messages, had_system_prompt))?;
                        model_used = model_ref.raw.clone();
                        let provider = crate::providers::resolve_provider(&model_ref);
                        let normalized = provider
                            .parse_chat_response(&val)
                            .map_err(|e| {
                                fail_partial(
                                    ChatError::Http(HttpError::InvalidJson(e.to_string())),
                                    &messages,
                                    had_system_prompt,
                                )
                            })?;
                        let retry_text = normalized.content.clone().unwrap_or_default();
                        let (out2, gn2) = guardrails.run_output(&retry_text).await;
                        match out2 {
                            GuardrailOutcome::Allow(t) => {
                                let retry_msg = ChatMessage {
                                    role: "assistant".to_string(),
                                    content: Some(MessageContent::Text(retry_text.clone())),
                                    tool_calls: None,
                                    tool_call_id: None,
                                    name: None,
                                    refusal: None,
                                };
                                messages.push(terminal_assistant_for_history(
                                    &retry_msg,
                                    &Some(t.clone()),
                                ));
                                let client_messages =
                                    conversation_messages_for_client(&messages, had_system_prompt);
                                final_outcome = Some(CompletionOutcome {
                                    content: Some(t),
                                    rounds: api_calls,
                                    usage,
                                    finish_reason: saved_finish.clone(),
                                    request_id: request_id.clone(),
                                    messages: client_messages,
                                    model_used: model_used.clone(),
                                });
                                break;
                            }
                            GuardrailOutcome::Block(r2) => {
                                metrics::counter!(crate::telemetry::metrics::COMPLETIONS_ERRORS, "model" => options.model.clone(), "error_kind" => "guardrail").increment(1);
                                return Err(fail_partial(
                                    ChatError::Guardrail(GuardrailError::new(
                                        GuardrailStage::Output,
                                        gn2,
                                        r2,
                                    )),
                                    &messages,
                                    had_system_prompt,
                                ));
                            }
                        }
                    } else {
                        metrics::counter!(crate::telemetry::metrics::COMPLETIONS_ERRORS, "model" => options.model.clone(), "error_kind" => "guardrail").increment(1);
                        return Err(fail_partial(
                            ChatError::Guardrail(GuardrailError::new(
                                GuardrailStage::Output,
                                guard_name,
                                reason,
                            )),
                            &messages,
                            had_system_prompt,
                        ));
                    }
                }
            }
        }

        let default_messages = {
            if final_outcome.is_none() {
                messages.push(msg.clone());
            }
            conversation_messages_for_client(&messages, had_system_prompt)
        };
        break final_outcome.unwrap_or(CompletionOutcome {
            content: saved_content,
            rounds: api_calls,
            usage,
            finish_reason: saved_finish,
            request_id: request_id.clone(),
            messages: default_messages,
            model_used: model_used.clone(),
        });
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!(crate::telemetry::metrics::COMPLETION_DURATION_MS, "model" => outcome.model_used.clone()).record(elapsed_ms);
    tracing::info!(
        request_id = %outcome.request_id,
        model = %outcome.model_used,
        rounds = outcome.rounds,
        elapsed_ms,
        "complete_with_tools finished"
    );

    Ok(outcome)
}

// ---------------------------------------------------------------------------
// Streaming chat completion (SSE / stream: true)
// ---------------------------------------------------------------------------

/// Outcome of a streaming completion (no tool-call loop).
#[derive(Debug, Clone)]
pub struct StreamOutcome {
    /// Full accumulated assistant content.
    pub content: String,
    /// `finish_reason` from the final chunk.
    pub finish_reason: Option<String>,
    /// Usage reported by the final chunk (only when `stream_options.include_usage = true`).
    pub usage: Option<proto::Usage>,
    /// Correlation ID for this request (UUID v4 auto-generated if not supplied by caller).
    pub request_id: String,
}

/// Stream a chat completion, calling `on_delta` for each content token as it arrives.
///
/// Unlike [`complete_with_tools`], this function does **not** execute tool calls.
/// Input guardrails run on the last user message before the stream starts; output
/// guardrails run on the fully-accumulated response after the stream ends (no retry —
/// same behaviour as gluellm's simple streaming path).
#[instrument(
    skip(http, hooks, guardrails, messages, options, on_delta),
    fields(model = %options.model)
)]
pub async fn stream_complete<F>(
    http: &HttpClient,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    messages: Vec<ChatMessage>,
    options: &ChatOptions,
    mut on_delta: F,
) -> Result<StreamOutcome, ChatError>
where
    F: FnMut(String) + Send,
{
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    tracing::info!(
        request_id = %request_id,
        model = %options.model,
        "stream_complete started"
    );

    let mut full_messages: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        full_messages.push(ChatMessage::text("system", sp));
    }
    full_messages.extend(messages);

    // --- Input guardrails ---
    if !guardrails.input_is_empty().await {
        let last_user = full_messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();

        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user {
                    if let Some(msg) = full_messages.iter_mut().rev().find(|m| m.role == "user") {
                        msg.content = Some(MessageContent::Text(transformed));
                    }
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(ChatError::Guardrail(GuardrailError::new(
                    GuardrailStage::Input,
                    guard_name,
                    reason,
                )));
            }
        }
    }

    // --- PreCompletion hook (observation only) ---
    let last_user = full_messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text().map(str::to_string))
        .unwrap_or_default();
    hooks
        .run(
            HookStage::PreCompletion,
            HookContext::new(HookStage::PreCompletion, last_user),
        )
        .await?;

    // Check cancellation before initiating the stream.
    if let Some(token) = &options.cancel {
        if token.is_cancelled() {
            return Err(ChatError::Cancelled);
        }
    }

    let credentials = credentials_for(options);
    let model_ref = crate::providers::parse_model_ref(&options.model);
    credentials
        .key_for(model_ref.provider)
        .map_err(ChatError::Credentials)?;
    let provider = crate::providers::resolve_provider(&model_ref);
    let ctx = crate::providers::ProviderRequestContext {
        model_ref: &model_ref,
        credentials: &credentials,
        messages: &full_messages,
        tools: None,
        chat_tools: None,
        stream: true,
        options,
    };
    let provider_req = provider.build_chat_request(&ctx);
    let header_refs: Vec<(&str, &str)> = provider_req
        .headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    tracing::debug!(%provider_req.url, "stream_complete POST (SSE)");

    let mut byte_stream = http
        .post_json_stream_with_headers(
            &provider_req.url,
            &provider_req.body,
            &header_refs,
            Some(provider_req.rate_limit_key),
        )
        .await?;
    tracing::debug!("stream_complete connection established, reading SSE chunks");

    let mut parser = SseParser::new();
    let mut outcome = StreamOutcome {
        content: String::new(),
        finish_reason: None,
        usage: None,
        request_id: request_id.clone(),
    };
    let mut byte_count = 0usize;
    let mut event_count = 0usize;

    while let Some(chunk) = byte_stream.next().await {
        // Check cancellation between chunks.
        if let Some(token) = &options.cancel {
            if token.is_cancelled() {
                return Err(ChatError::Cancelled);
            }
        }

        let bytes = chunk?;
        byte_count += bytes.len();
        let text = String::from_utf8_lossy(&bytes);
        tracing::trace!(
            chunk_bytes = bytes.len(),
            preview = ?&text[..text.len().min(120)],
            "stream_complete raw chunk"
        );

        let events = parser
            .push_str(&text)
            .map_err(|e| ChatError::Http(HttpError::InvalidJson(e.to_string())))?;

        for event in events {
            event_count += 1;
            if event.event.as_deref() == Some("error") {
                return Err(ChatError::Api(format!(
                    "stream error event: {}",
                    event.data.trim()
                )));
            }
            let data = event.data.trim();
            tracing::trace!(
                event_count,
                data_preview = ?&data[..data.len().min(80)],
                "stream_complete SSE event"
            );
            if data == "[DONE]" {
                tracing::trace!("stream_complete received [DONE]");
                break;
            }
            if data.is_empty() {
                continue;
            }
            let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        data_preview = ?&data[..data.len().min(200)],
                        "stream_complete JSON parse error"
                    );
                    return Err(ChatError::Serde(e));
                }
            };

            if let Some(u) = &chunk.usage {
                outcome.usage = Some(crate::usage::usage_from_breakdown(
                    crate::usage::breakdown_from_compat_usage(u),
                ));
            }

            for choice in &chunk.choices {
                if let Some(fr) = &choice.finish_reason {
                    tracing::trace!(finish_reason = %fr, "stream_complete finish_reason");
                    outcome.finish_reason = Some(fr.clone());
                }
                if let Some(delta) = &choice.delta.content
                    && !delta.is_empty()
                {
                    tracing::trace!(delta_len = delta.len(), "stream_complete delta");
                    outcome.content.push_str(delta);
                    on_delta(delta.clone());
                }
            }
        }
    }

    // --- Output guardrails (no retry for streaming) ---
    if !guardrails.output_is_empty().await {
        let (out_outcome, guard_name) = guardrails.run_output(&outcome.content).await;
        match out_outcome {
            GuardrailOutcome::Allow(transformed) => {
                outcome.content = transformed;
            }
            GuardrailOutcome::Block(reason) => {
                return Err(ChatError::Guardrail(GuardrailError::new(
                    GuardrailStage::Output,
                    guard_name,
                    reason,
                )));
            }
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics::histogram!(crate::telemetry::metrics::STREAM_DURATION_MS, "model" => options.model.clone()).record(elapsed_ms);

    tracing::info!(
        request_id = %outcome.request_id,
        model = %options.model,
        byte_count,
        event_count,
        content_chars = outcome.content.len(),
        elapsed_ms,
        "stream_complete finished"
    );

    Ok(outcome)
}

fn push_early_stream_tool<'a>(
    tc: ToolCall,
    in_flight: &mut FuturesUnordered<
        Pin<
            Box<
                dyn std::future::Future<Output = Result<(String, ChatMessage), ChatError>>
                    + Send
                    + 'a,
            >,
        >,
    >,
    dispatched_ids: &mut BTreeMap<String, ()>,
    hooks: &'a HookRegistry,
    registry: &'a ToolRegistry,
    status_emitter: Option<&'a Arc<StatusEmitter>>,
    request_id: &'a str,
    round: u32,
    model: &'a str,
    tool_result_max_chars: usize,
    loop_guard: &'a SharedToolLoopGuard,
) {
    if dispatched_ids.contains_key(&tc.id) {
        return;
    }
    dispatched_ids.insert(tc.id.clone(), ());
    let tool_id = tc.id.clone();
    let loop_guard = Arc::clone(loop_guard);
    in_flight.push(Box::pin(async move {
        let msg = dispatch_one(
            &tc,
            hooks,
            registry,
            status_emitter,
            request_id,
            round,
            model,
            true,
            tool_result_max_chars,
            Some(&loop_guard),
        )
        .await?;
        Ok((tool_id, msg))
    }));
}

/// Outcome of a streaming completion with tool rounds.
#[derive(Debug, Clone)]
pub struct StreamToolOutcome {
    pub content: String,
    pub finish_reason: Option<String>,
    pub usage: Option<proto::Usage>,
    pub request_id: String,
    pub rounds: u32,
    pub model_used: String,
    pub messages: Vec<ChatMessage>,
}

/// Stream a chat completion with multi-round tool execution (SSE).
///
/// Text deltas are forwarded to `on_delta`. Tool rounds mirror [`complete_with_tools`].
#[instrument(
    skip(http, registry, hooks, guardrails, caller_messages, options, on_delta),
    fields(model = %options.model)
)]
pub async fn stream_complete_with_tools<F>(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    caller_messages: Vec<ChatMessage>,
    options: &ChatOptions,
    mut on_delta: F,
) -> Result<StreamToolOutcome, ChatError>
where
    F: FnMut(String) + Send,
{
    let start = Instant::now();
    let request_id = options
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    tracing::info!(
        request_id = %request_id,
        model = %options.model,
        "stream_complete_with_tools started"
    );

    let credentials = credentials_for(options);

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    if let Some(sp) = &options.system_prompt {
        messages.push(ChatMessage::text("system", sp));
    }
    messages.extend(caller_messages);
    let had_system_prompt = options.system_prompt.is_some();

    if !guardrails.input_is_empty().await {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        let (outcome, guard_name) = guardrails.run_input(&last_user).await;
        match outcome {
            GuardrailOutcome::Allow(transformed) => {
                if transformed != last_user {
                    if let Some(msg) = messages.iter_mut().rev().find(|m| m.role == "user") {
                        msg.content = Some(MessageContent::Text(transformed));
                    }
                }
            }
            GuardrailOutcome::Block(reason) => {
                return Err(fail_partial(
                    ChatError::Guardrail(GuardrailError::new(
                        GuardrailStage::Input,
                        guard_name,
                        reason,
                    )),
                    &messages,
                    had_system_prompt,
                ));
            }
        }
    }

    let mut api_calls: u32 = 0;
    let mut model_used;
    let mut accumulated_usage: Option<proto::Usage> = None;
    // One automatic re-request when a provider returns a blank round (Groq flake).
    let mut empty_round_retries: u32 = 0;
    let loop_guard = new_tool_loop_guard();

    let all_specs = registry.list_specs().await;
    let mut active_set = ActiveToolSet::new(all_specs, options.tool_mode);
    let route_model = options
        .tool_route_model
        .as_deref()
        .unwrap_or(DEFAULT_TOOL_ROUTE_MODEL);

    loop {
        if api_calls >= options.max_tool_rounds {
            return Err(fail_partial(
                ChatError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                had_system_prompt,
            ));
        }
        api_calls += 1;

        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.as_ref())
            .and_then(|c| c.as_text().map(str::to_string))
            .unwrap_or_default();
        hooks
            .run(
                HookStage::PreCompletion,
                observation_hook_ctx(
                    HookStage::PreCompletion,
                    last_user.clone(),
                    &request_id,
                    api_calls,
                    &options.model,
                ),
            )
            .await
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, had_system_prompt))?;

        maybe_summarize_messages(
            http,
            &credentials,
            &mut messages,
            &options.summarize_context,
            options.aaak_compression_enabled,
            options.aaak_compression_model.as_deref(),
            &options.model,
            options,
            &request_id,
        )
        .await
        .map_err(|e| fail_partial(e, &messages, had_system_prompt))?;

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        notify_llm_payload(options, api_calls, &request_id, &messages);

        let tool_specs = active_set.specs_for_llm();
        let chat_tools = active_set.chat_tools_for_llm();

        let (mut byte_stream, model_ref) = provider_chat_stream(
            http,
            &credentials,
            &messages,
            tool_specs,
            chat_tools,
            options,
            &request_id,
            api_calls,
        )
        .await
        .map_err(|e| fail_partial(e, &messages, had_system_prompt))?;
        model_used = model_ref.raw.clone();
        let is_anthropic = model_ref.provider == crate::providers::ProviderId::Anthropic;

        let mut parser = SseParser::new();
        let mut round_content = String::new();
        let mut tool_dispatch = stream_tools::StreamingToolDispatch::new();
        let mut completed_tools: BTreeMap<String, ChatMessage> = BTreeMap::new();
        let mut dispatched_tool_ids: BTreeMap<String, ()> = BTreeMap::new();
        let mut in_flight: FuturesUnordered<
            Pin<
                Box<
                    dyn std::future::Future<Output = Result<(String, ChatMessage), ChatError>>
                        + Send,
                >,
            >,
        > = FuturesUnordered::new();
        let mut anthropic_acc =
            crate::providers::anthropic_stream::AnthropicStreamAccumulator::new();
        let mut round_finish: Option<String> = None;
        let mut round_usage: Option<proto::Usage> = None;
        let mut stream_done = false;

        while !stream_done || !in_flight.is_empty() {
            tokio::select! {
                biased;
                _ = async {
                    if let Some(token) = &options.cancel {
                        token.cancelled().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                }, if options.cancel.is_some() => {
                    return Err(fail_partial(
                        ChatError::Cancelled,
                        &messages,
                        had_system_prompt,
                    ));
                }
                result = in_flight.next(), if !in_flight.is_empty() => {
                    match result {
                        Some(Ok((id, msg))) => {
                            completed_tools.insert(id, msg);
                        }
                        Some(Err(e)) => {
                            return Err(fail_partial(e, &messages, had_system_prompt));
                        }
                        None => {}
                    }
                }
                chunk = byte_stream.next(), if !stream_done => {
                    let Some(chunk) = chunk else {
                        stream_done = true;
                        continue;
                    };
                    let bytes = chunk.map_err(|e| {
                        fail_partial(ChatError::from(e), &messages, had_system_prompt)
                    })?;
                    let text = String::from_utf8_lossy(&bytes);
                    let events = parser
                        .push_str(&text)
                        .map_err(|e| {
                            fail_partial(
                                ChatError::Http(HttpError::InvalidJson(e.to_string())),
                                &messages,
                                had_system_prompt,
                            )
                        })?;

                    for event in events {
                        if event.event.as_deref() == Some("error") {
                            return Err(fail_partial(
                                ChatError::Api(format!(
                                    "stream error event: {}",
                                    event.data.trim()
                                )),
                                &messages,
                                had_system_prompt,
                            ));
                        }
                        let data = event.data.trim();
                        if data == "[DONE]" {
                            stream_done = true;
                            break;
                        }
                        if data.is_empty() {
                            continue;
                        }
                        if is_anthropic {
                            if let Some(delta) = anthropic_acc
                                .apply_sse_data(data)
                                .map_err(|e| {
                                    fail_partial(ChatError::Serde(e), &messages, had_system_prompt)
                                })?
                                && !delta.is_empty()
                            {
                                round_content.push_str(&delta);
                                on_delta(delta);
                            }
                        } else {
                            let chunk: ChatCompletionChunk = serde_json::from_str(data)
                                .map_err(|e| {
                                    fail_partial(ChatError::Serde(e), &messages, had_system_prompt)
                                })?;
                            let prev_len = round_content.len();
                            let ready = stream_tools::apply_openai_chunk(
                                &chunk,
                                &mut round_content,
                                &mut tool_dispatch,
                                &mut round_finish,
                                &mut round_usage,
                            );
                            if round_content.len() > prev_len {
                                on_delta(round_content[prev_len..].to_string());
                            }
                            for tc in ready {
                                push_early_stream_tool(
                                    tc,
                                    &mut in_flight,
                                    &mut dispatched_tool_ids,
                                    hooks,
                                    registry,
                                    options.status_emitter.as_ref(),
                                    &request_id,
                                    api_calls,
                                    &options.model,
                                    options.tool_result_max_chars,
                                    &loop_guard,
                                );
                            }
                        }
                    }
                }
            }
        }

        if !is_anthropic {
            for tc in tool_dispatch.drain_at_round_end() {
                push_early_stream_tool(
                    tc,
                    &mut in_flight,
                    &mut dispatched_tool_ids,
                    hooks,
                    registry,
                    options.status_emitter.as_ref(),
                    &request_id,
                    api_calls,
                    &options.model,
                    options.tool_result_max_chars,
                    &loop_guard,
                );
            }
            while let Some(result) = in_flight.next().await {
                let (id, msg) = result.map_err(|e| fail_partial(e, &messages, had_system_prompt))?;
                completed_tools.insert(id, msg);
            }
        }

        let round = if is_anthropic {
            anthropic_acc.into_round_outcome()
        } else {
            let tool_calls = tool_dispatch.finish_remaining();
            crate::providers::StreamRoundOutcome {
                content: round_content,
                tool_calls,
                finish_reason: round_finish,
                usage: round_usage,
            }
        };

        let tool_call_count = round
            .tool_calls
            .iter()
            .filter(|tc| tc.kind == "function")
            .count() as u32;
        emit_safe(options.status_emitter.as_ref(), {
            let mut ev = ProcessEvent::new(ProcessEventKind::LlmCallEnd, &request_id, &model_used);
            ev.round = api_calls;
            ev.tool_call_count = tool_call_count;
            ev.usage = round.usage.clone();
            ev.estimated_cost_usd = round
                .usage
                .as_ref()
                .map(|u| estimate_model_call_cost_usd(&model_used, u));
            ev
        })
        .await;

        if let Some(ref u) = round.usage {
            accumulated_usage = Some(crate::usage::accumulate_usage(
                accumulated_usage.as_ref(),
                u,
            ));
        }

        hooks
            .run(
                HookStage::PostCompletion,
                observation_hook_ctx(
                    HookStage::PostCompletion,
                    round.content.clone(),
                    &request_id,
                    api_calls,
                    &model_used,
                ),
            )
            .await
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, had_system_prompt))?;

        // Groq (and some weaker models) intermittently finish with stop + empty
        // content and no tool_calls. Retry the same messages once before failing.
        if round.content.trim().is_empty() && round.tool_calls.is_empty() {
            if empty_round_retries < 1 {
                empty_round_retries += 1;
                // Don't burn a max_tool_rounds slot on a blank provider response.
                api_calls = api_calls.saturating_sub(1);
                tracing::warn!(
                    request_id = %request_id,
                    model = %model_used,
                    finish_reason = ?round.finish_reason,
                    "empty LLM round (no content, no tool calls) — retrying once"
                );
                continue;
            }
            tracing::warn!(
                request_id = %request_id,
                model = %model_used,
                finish_reason = ?round.finish_reason,
                "empty LLM round after retry — failing turn"
            );
            return Err(fail_partial(
                ChatError::EmptyResponse,
                &messages,
                had_system_prompt,
            ));
        }

        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: if round.content.is_empty() {
                None
            } else {
                Some(MessageContent::Text(round.content.clone()))
            },
            tool_calls: if round.tool_calls.is_empty() {
                None
            } else {
                Some(round.tool_calls.clone())
            },
            tool_call_id: None,
            name: None,
            refusal: None,
        };

        if !round.tool_calls.is_empty() {
            if active_set.has_router() && is_router_call(&round.tool_calls) {
                let query =
                    router_query_from_calls(&round.tool_calls).unwrap_or_else(|| last_user.clone());
                let user_context = user_context_for_route(&messages, &query);
                let matched = resolve_tool_route(
                    http,
                    &credentials,
                    &user_context,
                    active_set.dynamic_specs(),
                    route_model,
                    options,
                    &request_id,
                )
                .await;
                active_set.apply_route(matched);
                let matched_names: Vec<String> = active_set
                    .specs_for_llm()
                    .unwrap_or(&[])
                    .iter()
                    .filter(|s| !s.static_tool && s.name != crate::tools::ROUTER_TOOL_NAME)
                    .map(|s| s.name.clone())
                    .collect();
                emit_safe(options.status_emitter.as_ref(), {
                    let mut ev =
                        ProcessEvent::new(ProcessEventKind::ToolRoute, &request_id, route_model);
                    ev.round = api_calls;
                    ev.metadata.insert("route_query".to_string(), query.clone());
                    ev.metadata
                        .insert("matched_tools".to_string(), matched_names.join(","));
                    ev
                })
                .await;
                continue;
            }

            messages.push(msg.clone());

            // If the model's output was cut short by the token limit, the tool-call
            // arguments are truncated and cannot be parsed. Return a soft error for
            // each pending call so the model knows to retry with smaller arguments.
            if round.finish_reason.as_deref() == Some("length") {
                let function_tcs: Vec<_> = round
                    .tool_calls
                    .iter()
                    .filter(|tc| tc.kind == "function")
                    .collect();
                for tc in function_tcs {
                    let error_content = serde_json::to_string(&serde_json::json!({
                        "ok": false,
                        "error": format!(
                            "Your output was cut off by the model's token limit before the \
                             tool-call arguments were complete. The tool '{}' did not run. \
                             Please retry using smaller arguments — write one file at a time, \
                             use edit_file for large changes, or split large content into \
                             multiple smaller write_file calls.",
                            tc.function.name
                        )
                    }))
                    .unwrap_or_else(|_| r#"{"ok":false,"error":"output truncated"}"#.to_string());
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(MessageContent::Text(error_content)),
                        tool_calls: None,
                        tool_call_id: Some(tc.id.clone()),
                        name: Some(tc.function.name.clone()),
                        refusal: None,
                    });
                }
                continue;
            }

            let function_tcs: Vec<_> = round
                .tool_calls
                .iter()
                .filter(|tc| tc.kind == "function")
                .collect();
            if completed_tools.len() == function_tcs.len() && !function_tcs.is_empty() {
                for tc in function_tcs {
                    if let Some(msg) = completed_tools.remove(&tc.id) {
                        messages.push(msg);
                    }
                }
            } else {
                let results = futures_util::future::join_all(function_tcs.iter().map(|tc| {
                    dispatch_one(
                        tc,
                        hooks,
                        registry,
                        options.status_emitter.as_ref(),
                        &request_id,
                        api_calls,
                        &options.model,
                        true,
                        options.tool_result_max_chars,
                        Some(&loop_guard),
                    )
                }))
                .await;
                for r in results {
                    messages.push(r.map_err(|e| fail_partial(e, &messages, had_system_prompt))?);
                }
            }

            if options.condense_tool_messages {
                condense_tool_round(
                    &mut messages,
                    options.aaak_tool_condensing,
                    options.context_block_provider.as_ref(),
                    options.context_event_logger.as_ref(),
                );
            }
            continue;
        }

        let mut final_content = round.content;
        let final_finish = round.finish_reason;
        let final_usage = accumulated_usage;
        messages.push(msg.clone());

        if !guardrails.output_is_empty().await {
            let (out_outcome, guard_name) = guardrails.run_output(&final_content).await;
            match out_outcome {
                GuardrailOutcome::Allow(transformed) => {
                    final_content = transformed;
                }
                GuardrailOutcome::Block(reason) => {
                    return Err(fail_partial(
                        ChatError::Guardrail(GuardrailError::new(
                            GuardrailStage::Output,
                            guard_name,
                            reason,
                        )),
                        &messages,
                        had_system_prompt,
                    ));
                }
            }
        }

        let client_messages = conversation_messages_for_client(&messages, had_system_prompt);
        let outcome = StreamToolOutcome {
            content: final_content,
            finish_reason: final_finish,
            usage: final_usage,
            request_id: request_id.clone(),
            rounds: api_calls,
            model_used,
            messages: client_messages,
        };

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            request_id = %outcome.request_id,
            model = %outcome.model_used,
            rounds = outcome.rounds,
            elapsed_ms,
            "stream_complete_with_tools finished"
        );
        return Ok(outcome);
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::truncate_tool_result;
    use crate::openai::{ChatMessage, MessageContent};

    #[test]
    fn truncate_tool_result_appends_marker() {
        let content = "x".repeat(100);
        let out = truncate_tool_result(content.clone(), 50);
        assert!(out.contains("…\n[truncated]"));
        assert!(out.len() < content.len());
    }

    #[test]
    fn truncate_tool_result_zero_disables() {
        let content = "x".repeat(10_000);
        let out = truncate_tool_result(content.clone(), 0);
        assert_eq!(out, content);
    }

    #[test]
    fn fail_partial_wraps_cause_with_messages() {
        let messages = vec![
            ChatMessage::text("user", "hello"),
            ChatMessage::text("assistant", "partial"),
        ];
        let err = super::fail_partial(
            super::ChatError::MaxToolRounds(3),
            &messages,
            false,
        );
        let partial = err.partial_messages().expect("partial messages");
        assert_eq!(partial.len(), 2);
        assert!(matches!(err.root_cause(), super::ChatError::MaxToolRounds(3)));
        assert_eq!(err.to_string(), "exceeded max tool rounds (3)");
    }
}
