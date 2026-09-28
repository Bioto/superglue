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

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub use conversation::Conversation;

use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use secrecy::ExposeSecret;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::pin::Pin;
use thiserror::Error;
use tokio::time::{Duration, sleep};
use tracing::{instrument, warn};

use std::time::Instant;

use self::context_ops::{resolve_tool_route, user_context_for_route};
use crate::cancel::CancellationToken;
use crate::context::SummarizeContextConfig;
use crate::costing::apply_resolved_cost_usd;
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
    ActiveToolSet, CODE_TOOL_NAME, DEFAULT_TOOL_ROUTE_MODEL, OnToolError, ToolInvokeError,
    ToolMode, ToolRegistry, ToolRetryPolicy, is_router_call, router_call_id_from_calls,
    router_query_from_calls,
};
#[cfg(feature = "code")]
use crate::tools::{CodeLimits, source_from_arguments};

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

/// One section of the system prompt. Anthropic uses [`SystemPromptBlock::cache`] for
/// `cache_control: { type: "ephemeral" }`; OpenAI/xAI/Groq benefit from stable ordering.
#[derive(Debug, Clone)]
pub struct SystemPromptBlock {
    pub text: String,
    /// When true, Anthropic requests mark this block (and preceding tools) cacheable.
    pub cache: bool,
}

impl SystemPromptBlock {
    #[must_use]
    pub fn cached(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache: true,
        }
    }

    #[must_use]
    pub fn uncached(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache: false,
        }
    }
}

/// Whether [`ChatOptions`] include a configured system prompt (legacy or blocks).
#[must_use]
#[cfg(test)]
pub(crate) fn has_system_prompt(options: &ChatOptions) -> bool {
    options.system_prompt.is_some()
        || options
            .system_prompt_blocks
            .as_ref()
            .is_some_and(|blocks| !blocks.is_empty())
}

/// Concatenated system prompt for diagnostics and Responses `instructions`.
#[must_use]
pub(crate) fn effective_system_prompt(options: &ChatOptions) -> Option<String> {
    if let Some(blocks) = &options.system_prompt_blocks {
        let joined = blocks
            .iter()
            .map(|b| b.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if joined.is_empty() {
            None
        } else {
            Some(joined)
        }
    } else {
        options.system_prompt.clone()
    }
}

/// Count of leading `system` rows [`prepend_system_messages`] will inject.
#[must_use]
pub(crate) fn injected_system_prefix_count(options: &ChatOptions) -> usize {
    if let Some(blocks) = &options.system_prompt_blocks {
        blocks.iter().filter(|b| !b.text.is_empty()).count()
    } else if options
        .system_prompt
        .as_ref()
        .is_some_and(|s| !s.is_empty())
    {
        1
    } else {
        0
    }
}

/// Count of trailing `system` rows [`append_volatile_suffix`] will inject.
#[must_use]
pub(crate) fn injected_system_suffix_count(options: &ChatOptions) -> usize {
    options.volatile_suffix_blocks.as_ref().map_or(0, |blocks| {
        blocks.iter().filter(|b| !b.text.is_empty()).count()
    })
}

/// Prepend configured system blocks. Returns the number of rows injected.
pub(crate) fn prepend_system_messages(
    options: &ChatOptions,
    messages: &mut Vec<ChatMessage>,
) -> usize {
    let before = messages.len();
    if let Some(blocks) = &options.system_prompt_blocks {
        for block in blocks {
            if !block.text.is_empty() {
                messages.push(ChatMessage::text("system", &block.text));
            }
        }
    } else if let Some(sp) = &options.system_prompt
        && !sp.is_empty()
    {
        messages.push(ChatMessage::text("system", sp));
    }
    messages.len() - before
}

/// Append volatile system blocks after the conversation. Returns the number of rows injected.
pub(crate) fn append_volatile_suffix(
    options: &ChatOptions,
    messages: &mut Vec<ChatMessage>,
) -> usize {
    let before = messages.len();
    if let Some(blocks) = &options.volatile_suffix_blocks {
        for block in blocks {
            if !block.text.is_empty() {
                messages.push(ChatMessage::text("system", &block.text));
            }
        }
    }
    messages.len() - before
}

pub(crate) fn messages_with_volatile_suffix<'a>(
    options: &ChatOptions,
    messages: &'a [ChatMessage],
) -> Cow<'a, [ChatMessage]> {
    if injected_system_suffix_count(options) == 0 {
        Cow::Borrowed(messages)
    } else {
        let mut owned = messages.to_vec();
        append_volatile_suffix(options, &mut owned);
        Cow::Owned(owned)
    }
}

#[must_use]
pub fn prompt_prefix_hash(options: &ChatOptions, tool_names: &[&str]) -> u64 {
    let mut hasher = DefaultHasher::new();
    if let Some(blocks) = &options.system_prompt_blocks {
        for block in blocks {
            block.text.hash(&mut hasher);
        }
    } else if let Some(sp) = &options.system_prompt {
        sp.hash(&mut hasher);
    }
    for name in tool_names {
        name.hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn log_prefix_guard(
    options: &ChatOptions,
    tool_specs: Option<&[crate::tools::ToolSpec]>,
    last_hash: &mut Option<u64>,
) {
    let names: Vec<&str> = tool_specs
        .unwrap_or(&[])
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    let hash = prompt_prefix_hash(options, &names);
    if let (Some(logger), Some(prev)) = (&options.context_event_logger, *last_hash)
        && prev != hash
    {
        logger.log(format!(
            "context prefix_hash_changed prev={prev:016x} now={hash:016x} tools={}",
            names.len()
        ));
    }
    *last_hash = Some(hash);
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
            system_prompt: effective_system_prompt(options),
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
    /// Structured system sections (stable cached prefix first, dynamic tail after).
    /// When set, takes precedence over [`Self::system_prompt`] for message prepending.
    pub system_prompt_blocks: Option<Vec<SystemPromptBlock>>,
    /// Volatile system sections appended after conversation history (todos, memory).
    /// Excluded from returned client messages so they never persist into the next turn.
    pub volatile_suffix_blocks: Option<Vec<SystemPromptBlock>>,
    /// Sticky cache routing key (OpenAI `prompt_cache_key`, xAI Responses).
    pub prompt_cache_key: Option<String>,

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
    /// Warning: when true, tool-round condensing may rewrite the leading system
    /// row and destroy the prompt-cache prefix. Keep off unless you accept a miss.
    pub aaak_tool_condensing: bool,
    pub summarize_context: SummarizeContextConfig,
    pub aaak_compression_enabled: bool,
    pub aaak_compression_model: Option<String>,
    /// Optional compact context block appended after each condensed tool round.
    pub context_block_provider: Option<ContextBlockProvider>,
    /// Optional sink for context lifecycle diagnostics.
    pub context_event_logger: Option<ContextEventLogger>,

    /// Session image cache used to turn tool `vision.image_hash` results into multimodal input.
    pub image_store: Option<Arc<tokio::sync::Mutex<crate::images::ImageStore>>>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions {
            base_url: String::new(),
            api_key: secrecy::SecretString::from(String::new()),
            model: String::new(),
            max_tool_rounds: 0,
            system_prompt: None,
            system_prompt_blocks: None,
            volatile_suffix_blocks: None,
            prompt_cache_key: None,
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
            image_store: None,
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
            system_prompt_blocks: None,
            volatile_suffix_blocks: None,
            prompt_cache_key: None,
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
            reasoning_summary: p
                .reasoning_summary
                .as_deref()
                .and_then(reasoning::ReasoningSummaryLevel::parse_str)
                .unwrap_or_default(),
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
                keep_recent: usize::try_from(p.summarize_context_keep_recent.unwrap_or(6))
                    .unwrap_or(12),
                max_chars: 800_000,
                ..SummarizeContextConfig::default()
            },
            aaak_compression_enabled: p.aaak_compression_enabled.unwrap_or(false),
            aaak_compression_model: p.aaak_compression_model,
            context_block_provider: None,
            context_event_logger: None,
            image_store: None,
        }
    }
}

fn parse_tool_mode(s: Option<&str>) -> ToolMode {
    s.map(ToolMode::parse).unwrap_or_default()
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
    /// Excludes synthetic `system` rows injected from [`ChatOptions::system_prompt_blocks`]
    /// or [`ChatOptions::system_prompt`], and trailing [`ChatOptions::volatile_suffix_blocks`],
    /// so this slice can be passed back into [`complete_with_tools`] on the next turn.
    pub messages: Vec<ChatMessage>,
    /// Model that produced the final successful HTTP response (after any fallback).
    pub model_used: String,
}

/// Strips injected leading and trailing `system` rows.
///
/// Drops at most `injected_prefix` leading `system` rows and `injected_suffix`
/// trailing `system` rows. A caller-supplied `system` row after the injected
/// prefix is kept.
#[must_use]
pub fn conversation_messages_for_client(
    messages: &[ChatMessage],
    injected_prefix: usize,
    injected_suffix: usize,
) -> Vec<ChatMessage> {
    let mut start = 0;
    while start < injected_prefix && start < messages.len() && messages[start].role == "system" {
        start += 1;
    }
    let mut end = messages.len();
    let mut stripped = 0;
    while stripped < injected_suffix && end > start && messages[end - 1].role == "system" {
        end -= 1;
        stripped += 1;
    }
    messages[start..end].to_vec()
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

    /// True when the provider closed a stream before any useful tokens.
    /// Common on gateway idle cutoffs (`error decoding response body`) and HTTP timeouts.
    pub fn is_transient_stream_stall(&self) -> bool {
        detail_is_transient_stream_stall(&self.root_cause().to_string())
    }
}

pub(crate) fn resolve_chat_provider(
    model_ref: &crate::providers::ModelRef,
) -> Result<Box<dyn crate::providers::LlmProvider>, ChatError> {
    crate::providers::resolve_provider(model_ref)
        .map_err(|err| ChatError::UnsupportedProvider(err.0))
}

/// Shared classifier for stream stalls (timeouts, truncated SSE, connection reset).
pub fn detail_is_transient_stream_stall(detail: &str) -> bool {
    let detail = detail.to_ascii_lowercase();
    detail.contains("error decoding response body")
        || detail.contains("timed out")
        || detail.contains("operation timed out")
        || detail.contains("connection reset")
        || detail.contains("connection closed")
        || detail.contains("error sending request")
}

const MAX_STREAM_STALL_RETRIES: u32 = 2;

pub(crate) fn should_retry_stalled_stream_round(
    err: &ChatError,
    stall_retries: u32,
    round_content: &str,
    tools_started: bool,
) -> bool {
    stall_retries < MAX_STREAM_STALL_RETRIES
        && round_content.is_empty()
        && !tools_started
        && err.is_transient_stream_stall()
}

/// Wrap a tool-loop failure with the messages accumulated so far.
pub fn fail_partial(
    cause: ChatError,
    messages: &[ChatMessage],
    injected_prefix: usize,
) -> ChatError {
    ChatError::PartialTurn {
        cause: Box::new(cause),
        messages: conversation_messages_for_client(messages, injected_prefix, 0),
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

const OUTPUT_REVISE_TIMEOUT: Duration = Duration::from_millis(400);
const OUTPUT_REVISE_NOUL: f64 = 0.5;

/// True when an output block should burn another generation.
async fn output_block_should_regen(
    http: &HttpClient,
    options: &ChatOptions,
    text: &str,
    reason: &str,
) -> bool {
    use crate::providers::ProviderId;
    use crate::systemone::{Noul, Questions, SystemOneRequest, clip_state, noul_yes, system_one};

    let creds = credentials_for(options);
    if !creds.has_key(ProviderId::TypeSafe) {
        return true;
    }
    let mut questions = Questions::new();
    questions.insert(
        "regen",
        Noul::new("Is this a clear policy violation that needs another generation?").criteria(
            "Yes. The text must be rewritten.",
            "No. The block is a low-confidence false positive.",
        ),
    );
    let state = format!(
        "Policy reason: {}\n\nAssistant text: {}",
        clip_state(reason, 160),
        clip_state(text, 400)
    );
    let request = SystemOneRequest::new(state, questions).with_model("typesafe:jev-latest");
    let work = system_one(http, &creds, request);
    match tokio::time::timeout(OUTPUT_REVISE_TIMEOUT, work).await {
        Ok(Ok(response)) => response
            .answer("regen")
            .is_none_or(|answer| noul_yes(answer, OUTPUT_REVISE_NOUL)),
        _ => true,
    }
}

/// Perform one HTTP JSON POST, honouring an optional cancellation token.
pub(crate) async fn post_json_cancellable(
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

fn last_user_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|message| message.content.as_ref())
        .map(MessageContent::text_for_summary)
        .unwrap_or_default()
}

fn oi_params(options: &ChatOptions) -> crate::telemetry::openinference::InvocationParams<'_> {
    crate::telemetry::openinference::InvocationParams {
        temperature: options.temperature,
        top_p: options.top_p,
        max_completion_tokens: options.max_completion_tokens,
        reasoning_effort: options.reasoning_effort.as_deref(),
        seed: options.seed,
    }
}

fn finish_llm_chat(
    llm: &crate::telemetry::openinference::LlmSpan,
    provider: &dyn crate::providers::LlmProvider,
    model_ref: &crate::providers::ModelRef,
    body: &Value,
) {
    #[cfg(feature = "otlp")]
    {
        match provider.parse_chat_response(body) {
            Ok(normalized) => {
                let output = ChatMessage {
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
                    provider_blocks: None,
                };
                llm.finish(&crate::telemetry::openinference::LlmFinish {
                    model_name: &model_ref.model,
                    provider: model_ref.provider.as_str(),
                    finish_reason: normalized.finish_reason.as_deref(),
                    usage: normalized.usage.as_ref(),
                    output: &output,
                });
            }
            Err(err) => llm.fail(&err.to_string()),
        }
    }
    #[cfg(not(feature = "otlp"))]
    {
        let _ = (llm, provider, model_ref, body);
    }
}

fn finish_llm_round(
    llm: &crate::telemetry::openinference::LlmSpan,
    model_ref: &crate::providers::ModelRef,
    content: &str,
    tool_calls: &[ToolCall],
    finish_reason: Option<&str>,
    usage: Option<&proto::Usage>,
) {
    #[cfg(feature = "otlp")]
    {
        let output = ChatMessage {
            role: "assistant".to_string(),
            content: if content.is_empty() {
                None
            } else {
                Some(MessageContent::Text(content.to_string()))
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.to_vec())
            },
            tool_call_id: None,
            name: None,
            refusal: None,
            provider_blocks: None,
        };
        llm.finish(&crate::telemetry::openinference::LlmFinish {
            model_name: &model_ref.model,
            provider: model_ref.provider.as_str(),
            finish_reason,
            usage,
            output: &output,
        });
    }
    #[cfg(not(feature = "otlp"))]
    {
        let _ = (llm, model_ref, content, tool_calls, finish_reason, usage);
    }
}

fn begin_llm<'a>(
    session_id: &'a str,
    model_name: &'a str,
    provider: &'a str,
    messages: &'a [ChatMessage],
    tools: Option<&'a [crate::tools::ToolSpec]>,
    params: &'a crate::telemetry::openinference::InvocationParams<'a>,
) -> crate::telemetry::openinference::LlmSpan {
    crate::telemetry::openinference::LlmSpan::begin_chat(
        &crate::telemetry::openinference::LlmStart {
            session_id,
            model_name,
            provider,
            messages,
            tools,
            params,
        },
    )
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

        let provider = resolve_chat_provider(&model_ref)?;
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
        let params = oi_params(options);
        let llm = begin_llm(
            request_id,
            &model_ref.model,
            model_ref.provider.as_str(),
            messages,
            tool_specs,
            &params,
        );

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
            Ok(v) => {
                finish_llm_chat(&llm, provider.as_ref(), &model_ref, &v);
                return Ok((v, model_ref));
            }
            Err(e) => {
                llm.fail(&e.to_string());
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

        let provider = resolve_chat_provider(&model_ref)?;
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

/// Shared environment for [`dispatch_one`] and early-stream tool dispatch.
#[derive(Clone)]
pub(crate) struct DispatchCtx<'a> {
    pub hooks: &'a HookRegistry,
    pub registry: &'a ToolRegistry,
    pub status_emitter: Option<&'a Arc<StatusEmitter>>,
    pub request_id: &'a str,
    pub round: u32,
    pub model: &'a str,
    pub emit_start: bool,
    pub tool_result_max_chars: usize,
    pub loop_guard: Option<&'a SharedToolLoopGuard>,
    pub cancel: Option<&'a CancellationToken>,
    pub code_allowlist: Option<Arc<HashSet<String>>>,
}

/// Dispatch a single `"function"` tool call through the full pre/post hook pipeline.
///
/// Returns the [`ChatMessage`] with `role="tool"` that should be appended to the
/// conversation history. Called concurrently for all tool calls in a single round
/// via [`futures_util::future::join_all`].
#[instrument(
    skip(tc, ctx),
    fields(tool.name = %tc.function.name, tool.id = %tc.id)
)]
pub(crate) async fn dispatch_one(
    tc: &ToolCall,
    ctx: DispatchCtx<'_>,
) -> Result<ChatMessage, ChatError> {
    let DispatchCtx {
        hooks,
        registry,
        status_emitter,
        request_id,
        round,
        model,
        emit_start,
        tool_result_max_chars,
        loop_guard,
        cancel,
        code_allowlist,
    } = ctx;
    let tool_name = tc.function.name.clone();
    crate::telemetry::openinference::tag_tool(
        request_id,
        &tool_name,
        &tc.id,
        tc.function.arguments.trim(),
    );
    if emit_start && let Some(emitter) = status_emitter {
        let mut ev = ProcessEvent::new(ProcessEventKind::ToolCallStart, request_id, model);
        ev.round = round;
        ev.metadata
            .insert("tool_name".to_string(), tool_name.clone());
        ev.metadata
            .insert("tool_call_id".to_string(), tc.id.clone());
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
        if tc.function.name == CODE_TOOL_NAME {
            let Some(allowlist) = code_allowlist.as_deref() else {
                return Ok(serde_json::to_string(&json!({
                    "ok": false,
                    "error": "code is only available after request_tools in tool_mode=code"
                }))?);
            };
            #[cfg(not(feature = "code"))]
            {
                let _ = allowlist;
                return Ok(serde_json::to_string(&json!({
                    "ok": false,
                    "error": "tool mode `code` is unavailable in this build; enable SuperGlue feature `code`"
                }))?);
            }
            #[cfg(feature = "code")]
            {
            let source = match source_from_arguments(&args) {
                Ok(source) => source,
                Err(err) => {
                    if cancel.is_some_and(|token| token.is_cancelled()) {
                        return Err(ChatError::Cancelled);
                    }
                    return Ok(serde_json::to_string(&json!({
                        "ok": false,
                        "error": err.to_string(),
                    }))?);
                }
            };
            let value = match crate::tools::execute_code_with_cancel(
                &source,
                allowlist,
                registry,
                CodeLimits::default(),
                cancel,
            )
            .await
            {
                Ok(value) => value,
                Err(err) => {
                    if cancel.is_some_and(|token| token.is_cancelled()) {
                        return Err(ChatError::Cancelled);
                    }
                    return Ok(serde_json::to_string(&json!({
                        "ok": false,
                        "error": err.to_string(),
                    }))?);
                }
            };
            return Ok(serde_json::to_string(&value)?);
            }
        }

        let (tool, policy) = registry
            .resolve_invocation(&tc.function.name)
            .await
            .map_err(ChatError::Tool)?;

        if let Some(guard) = loop_guard {
            let mut g = guard.lock().await;
            if let Some(suppressed) = g.check_suppressed(&tc.function.name, &args) {
                return Ok(suppressed);
            }
        }

        // Serialize same-file mutations so concurrent edits in one round cannot
        // invalidate each other's anchors.
        let _file_guard = file_locks::acquire_file_lock(&tc.function.name, &args).await;

        let result = match cancel {
            Some(token) => {
                tokio::select! {
                    biased;
                    () = token.cancelled() => return Err(ChatError::Cancelled),
                    result = invoke_with_policy(
                        &tool,
                        &tc.function.name,
                        args.clone(),
                        &policy,
                    ) => result?,
                }
            }
            None => invoke_with_policy(&tool, &tc.function.name, args.clone(), &policy).await?,
        };
        let result_json = serde_json::to_string(&result)?;

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
        ev.metadata
            .insert("tool_call_id".to_string(), tc.id.clone());
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

    match &exec_result {
        Ok(content) => crate::telemetry::openinference::set_output_text(content),
        Err(err) => crate::telemetry::openinference::fail_current(&err.to_string()),
    }

    let content = truncate_tool_result(exec_result?, tool_result_max_chars);
    Ok(ChatMessage {
        role: "tool".to_string(),
        content: Some(MessageContent::Text(content)),
        tool_calls: None,
        tool_call_id: Some(tc.id.clone()),
        name: Some(tool_name),
        refusal: None,
        provider_blocks: None,
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
    crate::telemetry::openinference::tag_chain(&request_id);
    metrics::counter!(crate::telemetry::metrics::COMPLETIONS_TOTAL, "model" => options.model.clone()).increment(1);

    let credentials = credentials_for(options);

    // Prepend system prompt if configured.
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    let injected_prefix = prepend_system_messages(options, &mut messages);
    messages.extend(caller_messages);
    crate::telemetry::openinference::set_input_text(&last_user_text(&messages));

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
                    injected_prefix,
                ));
            }
        }
    }

    let mut api_calls: u32 = 0;
    let mut last_prompt_tokens: Option<u32> = None;
    let mut model_used;
    let mut accumulated_usage: Option<proto::Usage> = None;
    // One automatic re-request when a provider returns a blank round (Groq flake).
    let mut empty_round_retries: u32 = 0;
    let loop_guard = new_tool_loop_guard();
    let mut last_prefix_hash = None;

    let all_specs = registry.list_specs().await;
    let mut active_set = ActiveToolSet::new(all_specs, options.tool_mode);
    let route_model = options
        .tool_route_model
        .as_deref()
        .unwrap_or(DEFAULT_TOOL_ROUTE_MODEL);

    if let Some(token) = &options.cancel
        && token.is_cancelled()
    {
        return Err(ChatError::Cancelled);
    }

    let outcome = loop {
        if api_calls >= options.max_tool_rounds {
            metrics::counter!(crate::telemetry::metrics::COMPLETIONS_ERRORS, "model" => options.model.clone(), "error_kind" => "max_tool_rounds").increment(1);
            return Err(fail_partial(
                ChatError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                injected_prefix,
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
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, injected_prefix))?;

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
            last_prompt_tokens,
        )
        .await
        .map_err(|e| fail_partial(e, &messages, injected_prefix))?;

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        let tool_specs = active_set.specs_for_llm();
        let chat_tools = active_set.chat_tools_for_llm();
        log_prefix_guard(options, tool_specs, &mut last_prefix_hash);
        let request_messages = messages_with_volatile_suffix(options, &messages);
        notify_llm_payload(options, api_calls, &request_id, &request_messages);

        let (val, model_ref) = match provider_chat_post(
            http,
            &credentials,
            &request_messages,
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
                return Err(fail_partial(e, &messages, injected_prefix));
            }
        };
        model_used = model_ref.raw.clone();
        let provider = resolve_chat_provider(&model_ref)
            .map_err(|e| fail_partial(e, &messages, injected_prefix))?;
        let normalized = provider.parse_chat_response(&val).map_err(|e| {
            fail_partial(
                if e.to_string().contains("no choices") {
                    ChatError::NoChoice
                } else {
                    ChatError::Http(HttpError::InvalidJson(e.to_string()))
                },
                &messages,
                injected_prefix,
            )
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
            provider_blocks: normalized.provider_blocks.clone(),
        };

        let tool_call_count = normalized
            .tool_calls
            .iter()
            .filter(|tc| tc.kind == "function")
            .count() as u32;
        let mut usage_proto = normalized.usage.clone();
        if let Some(ref u) = usage_proto {
            if u.prompt_tokens > 0 {
                last_prompt_tokens = Some(u.prompt_tokens);
            }
            accumulated_usage = Some(crate::usage::accumulate_usage(
                accumulated_usage.as_ref(),
                u,
            ));
        }
        let estimated_cost = usage_proto
            .as_mut()
            .map(|u| apply_resolved_cost_usd(&model_used, u, u.cost_usd));

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
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, injected_prefix))?;

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
                let catalog = (active_set.mode() == ToolMode::Code)
                    .then(|| active_set.route_catalog(&matched));
                let router_call_id = router_call_id_from_calls(&normalized.tool_calls);
                active_set.apply_route(matched);
                let matched_names: Vec<String> = if active_set.mode() == ToolMode::Code {
                    active_set.code_allowlist().into_iter().collect()
                } else {
                    active_set
                        .specs_for_llm()
                        .unwrap_or(&[])
                        .iter()
                        .filter(|s| !s.static_tool && s.name != crate::tools::ROUTER_TOOL_NAME)
                        .map(|s| s.name.clone())
                        .collect()
                };
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
                if let Some(catalog) = catalog {
                    messages.push(msg.clone());
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(MessageContent::Text(
                            serde_json::to_string(&catalog).unwrap_or_else(|_| "{}".into()),
                        )),
                        tool_calls: None,
                        tool_call_id: router_call_id,
                        name: Some(crate::tools::ROUTER_TOOL_NAME.to_string()),
                        refusal: None,
                        provider_blocks: None,
                    });
                }
                continue;
            }

            messages.push(msg.clone());

            // Run tool handlers concurrently on this process (order of `tool` messages follows `tool_calls`).
            tracing::info!(
                count = function_tcs.len(),
                request_id = %request_id,
                "tool_calls_batch"
            );
            let code_allowlist = (active_set.mode() == ToolMode::Code && active_set.routed)
                .then(|| Arc::new(active_set.code_allowlist()));
            let ctx = DispatchCtx {
                hooks,
                registry,
                status_emitter: options.status_emitter.as_ref(),
                request_id: &request_id,
                round: api_calls,
                model: &options.model,
                emit_start: true,
                tool_result_max_chars: options.tool_result_max_chars,
                loop_guard: Some(&loop_guard),
                cancel: options.cancel.as_ref(),
                code_allowlist,
            };
            let results = futures_util::future::join_all(
                function_tcs.iter().map(|tc| dispatch_one(tc, ctx.clone())),
            )
            .await;
            for r in results {
                messages.push(r.map_err(|e| fail_partial(e, &messages, injected_prefix))?);
            }

            if let Some(store) = &options.image_store {
                let store = store.lock().await;
                let _ = crate::images::attach_vision_from_tool_results(&mut messages, &store);
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
                injected_prefix,
            ));
        }

        let usage = accumulated_usage.clone();
        let content = normalized.content.clone();
        let finish_reason = normalized.finish_reason.clone();

        // --- Output guardrails: retry loop ---
        if guardrails.output_is_empty().await {
            messages.push(msg.clone());
            let client_messages = conversation_messages_for_client(&messages, injected_prefix, 0);
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
                        conversation_messages_for_client(&messages, injected_prefix, 0);
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
                    if !output_block_should_regen(http, options, &text, &reason).await {
                        warn!(
                            guard = %guard_name,
                            "output guardrail block skipped regen"
                        );
                        messages.push(terminal_assistant_for_history(&msg, &saved_content));
                        let client_messages =
                            conversation_messages_for_client(&messages, injected_prefix, 0);
                        final_outcome = Some(CompletionOutcome {
                            content: saved_content.clone(),
                            rounds: api_calls,
                            usage,
                            finish_reason: saved_finish.clone(),
                            request_id: request_id.clone(),
                            messages: client_messages,
                            model_used: model_used.clone(),
                        });
                        break;
                    }
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
                                injected_prefix,
                            ));
                        }
                        api_calls += 1;
                        let credentials = credentials_for(options);
                        let request_messages = messages_with_volatile_suffix(options, &messages);
                        let (val, model_ref) = provider_chat_post(
                            http,
                            &credentials,
                            &request_messages,
                            None,
                            None,
                            options,
                            &request_id,
                            api_calls,
                            false,
                        )
                        .await
                        .map_err(|e| fail_partial(e, &messages, injected_prefix))?;
                        model_used = model_ref.raw.clone();
                        let provider = resolve_chat_provider(&model_ref)
                            .map_err(|e| fail_partial(e, &messages, injected_prefix))?;
                        let normalized = provider.parse_chat_response(&val).map_err(|e| {
                            fail_partial(
                                ChatError::Http(HttpError::InvalidJson(e.to_string())),
                                &messages,
                                injected_prefix,
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
                                    provider_blocks: None,
                                };
                                messages.push(terminal_assistant_for_history(
                                    &retry_msg,
                                    &Some(t.clone()),
                                ));
                                let client_messages =
                                    conversation_messages_for_client(&messages, injected_prefix, 0);
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
                                    injected_prefix,
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
                            injected_prefix,
                        ));
                    }
                }
            }
        }

        let default_messages = {
            if final_outcome.is_none() {
                messages.push(msg.clone());
            }
            conversation_messages_for_client(&messages, injected_prefix, 0)
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
    crate::telemetry::openinference::set_output_text(outcome.content.as_deref().unwrap_or(""));

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
    crate::telemetry::openinference::tag_chain(&request_id);

    let mut full_messages: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    let _injected_prefix = prepend_system_messages(options, &mut full_messages);
    full_messages.extend(messages);
    crate::telemetry::openinference::set_input_text(&last_user_text(&full_messages));

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
    let provider = resolve_chat_provider(&model_ref)?;
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

    let params = oi_params(options);
    let llm = begin_llm(
        &request_id,
        &model_ref.model,
        model_ref.provider.as_str(),
        &full_messages,
        None,
        &params,
    );
    let mut byte_stream = match http
        .post_json_stream_with_headers(
            &provider_req.url,
            &provider_req.body,
            &header_refs,
            Some(provider_req.rate_limit_key),
        )
        .await
    {
        Ok(stream) => stream,
        Err(err) => {
            llm.fail(&err.to_string());
            return Err(ChatError::Http(err));
        }
    };
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
                outcome.usage = Some(crate::usage::usage_from_compat(u));
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
                if let Some(delta) = choice.delta.reasoning_text() {
                    emit_reasoning_delta(
                        options.status_emitter.as_ref(),
                        &outcome.request_id,
                        &options.model,
                        1,
                        &delta,
                    )
                    .await;
                }
            }
        }
    }

    finish_llm_round(
        &llm,
        &model_ref,
        &outcome.content,
        &[],
        outcome.finish_reason.as_deref(),
        outcome.usage.as_ref(),
    );

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
    crate::telemetry::openinference::set_output_text(&outcome.content);

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
    ctx: DispatchCtx<'a>,
) {
    if dispatched_ids.contains_key(&tc.id) {
        return;
    }
    dispatched_ids.insert(tc.id.clone(), ());
    let tool_id = tc.id.clone();
    in_flight.push(Box::pin(async move {
        let msg = dispatch_one(&tc, ctx).await?;
        Ok((tool_id, msg))
    }));
}

async fn emit_reasoning_delta(
    emitter: Option<&Arc<StatusEmitter>>,
    request_id: &str,
    model: &str,
    round: u32,
    delta: &str,
) {
    if delta.is_empty() {
        return;
    }
    let mut ev = ProcessEvent::new(ProcessEventKind::ReasoningDelta, request_id, model);
    ev.round = round;
    ev.metadata.insert("delta".to_string(), delta.to_string());
    emit_safe(emitter, ev).await;
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
/// Text deltas are forwarded to `on_delta`. Reasoning text goes to
/// `on_reasoning_delta`. Tool rounds mirror [`complete_with_tools`].
#[instrument(
    skip(http, registry, hooks, guardrails, caller_messages, options, on_delta, on_reasoning_delta),
    fields(model = %options.model)
)]
pub async fn stream_complete_with_tools<FO, FR>(
    http: &HttpClient,
    registry: &ToolRegistry,
    hooks: &HookRegistry,
    guardrails: &GuardrailRegistry,
    caller_messages: Vec<ChatMessage>,
    options: &ChatOptions,
    mut on_delta: FO,
    mut on_reasoning_delta: FR,
) -> Result<StreamToolOutcome, ChatError>
where
    FO: FnMut(String) + Send,
    FR: FnMut(String) + Send,
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
    crate::telemetry::openinference::tag_chain(&request_id);

    let credentials = credentials_for(options);

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(caller_messages.len() + 1);
    let injected_prefix = prepend_system_messages(options, &mut messages);
    messages.extend(caller_messages);
    crate::telemetry::openinference::set_input_text(&last_user_text(&messages));

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
                    injected_prefix,
                ));
            }
        }
    }

    let mut api_calls: u32 = 0;
    let mut last_prompt_tokens: Option<u32> = None;
    let mut model_used;
    let mut accumulated_usage: Option<proto::Usage> = None;
    // One automatic re-request when a provider returns a blank round (Groq flake).
    let mut empty_round_retries: u32 = 0;
    // Re-request the same round when the HTTP stream dies before the first token.
    let mut stall_retries: u32 = 0;
    let loop_guard = new_tool_loop_guard();
    let mut last_prefix_hash = None;

    let all_specs = registry.list_specs().await;
    let mut active_set = ActiveToolSet::new(all_specs, options.tool_mode);
    let mut stream_code_allowlist: Option<Arc<HashSet<String>>> = None;
    let route_model = options
        .tool_route_model
        .as_deref()
        .unwrap_or(DEFAULT_TOOL_ROUTE_MODEL);

    loop {
        if api_calls >= options.max_tool_rounds {
            return Err(fail_partial(
                ChatError::MaxToolRounds(options.max_tool_rounds),
                &messages,
                injected_prefix,
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
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, injected_prefix))?;

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
            last_prompt_tokens,
        )
        .await
        .map_err(|e| fail_partial(e, &messages, injected_prefix))?;

        emit_safe(options.status_emitter.as_ref(), {
            let mut ev =
                ProcessEvent::new(ProcessEventKind::LlmCallStart, &request_id, &options.model);
            ev.round = api_calls;
            ev
        })
        .await;

        let tool_specs = active_set.specs_for_llm();
        let chat_tools = active_set.chat_tools_for_llm();
        log_prefix_guard(options, tool_specs, &mut last_prefix_hash);
        let request_messages = messages_with_volatile_suffix(options, &messages);
        notify_llm_payload(options, api_calls, &request_id, &request_messages);

        let requested = crate::providers::parse_model_ref(&options.model);
        let params = oi_params(options);
        let llm = begin_llm(
            &request_id,
            &requested.model,
            requested.provider.as_str(),
            &request_messages,
            tool_specs,
            &params,
        );
        let (mut byte_stream, model_ref) = match provider_chat_stream(
            http,
            &credentials,
            &request_messages,
            tool_specs,
            chat_tools,
            options,
            &request_id,
            api_calls,
        )
        .await
        {
            Ok(stream) => stream,
            Err(e) => {
                llm.fail(&e.to_string());
                if should_retry_stalled_stream_round(&e, stall_retries, "", false) {
                    stall_retries += 1;
                    api_calls = api_calls.saturating_sub(1);
                    tracing::warn!(
                        request_id = %request_id,
                        model = %options.model,
                        attempt = stall_retries,
                        error = %e,
                        "LLM stream failed before first token — retrying round"
                    );
                    sleep(Duration::from_millis(400)).await;
                    continue;
                }
                return Err(fail_partial(e, &messages, injected_prefix));
            }
        };
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
        let mut dispatch_ctx = DispatchCtx {
            hooks,
            registry,
            status_emitter: options.status_emitter.as_ref(),
            request_id: &request_id,
            round: api_calls,
            model: &options.model,
            emit_start: true,
            tool_result_max_chars: options.tool_result_max_chars,
            loop_guard: Some(&loop_guard),
            cancel: options.cancel.as_ref(),
            code_allowlist: stream_code_allowlist.clone(),
        };
        let mut anthropic_acc =
            crate::providers::anthropic_stream::AnthropicStreamAccumulator::new();
        let mut round_finish: Option<String> = None;
        let mut round_usage: Option<proto::Usage> = None;
        let mut stream_done = false;
        let mut retry_stalled_round = false;

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
                        injected_prefix,
                    ));
                }
                result = in_flight.next(), if !in_flight.is_empty() => {
                    match result {
                        Some(Ok((id, msg))) => {
                            completed_tools.insert(id, msg);
                        }
                        Some(Err(e)) => {
                            return Err(fail_partial(e, &messages, injected_prefix));
                        }
                        None => {}
                    }
                }
                chunk = byte_stream.next(), if !stream_done => {
                    let Some(chunk) = chunk else {
                        stream_done = true;
                        continue;
                    };
                    let bytes = match chunk {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            let chat_err = ChatError::from(e);
                            let tools_started = !completed_tools.is_empty()
                                || !dispatched_tool_ids.is_empty()
                                || !in_flight.is_empty();
                            if should_retry_stalled_stream_round(
                                &chat_err,
                                stall_retries,
                                &round_content,
                                tools_started,
                            ) {
                                retry_stalled_round = true;
                                stream_done = true;
                            } else {
                                return Err(fail_partial(
                                    chat_err,
                                    &messages,
                                    injected_prefix,
                                ));
                            }
                            continue;
                        }
                    };
                    let text = String::from_utf8_lossy(&bytes);
                    let events = parser
                        .push_str(&text)
                        .map_err(|e| {
                            fail_partial(
                                ChatError::Http(HttpError::InvalidJson(e.to_string())),
                                &messages,
                                injected_prefix,
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
                                injected_prefix,
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
                                    fail_partial(ChatError::Serde(e), &messages, injected_prefix)
                                })?
                            {
                                match delta {
                                    crate::providers::anthropic_stream::AnthropicStreamDelta::Text(
                                        t,
                                    ) => {
                                        round_content.push_str(&t);
                                        on_delta(t);
                                    }
                                    crate::providers::anthropic_stream::AnthropicStreamDelta::Thinking(
                                        t,
                                    ) => {
                                        on_reasoning_delta(t.clone());
                                        emit_reasoning_delta(
                                            options.status_emitter.as_ref(),
                                            &request_id,
                                            &options.model,
                                            api_calls,
                                            &t,
                                        )
                                        .await;
                                    }
                                }
                            }
                            // Anthropic never sends `[DONE]`; stop at `message_stop`
                            // rather than waiting for the socket to close.
                            if crate::providers::anthropic_stream::AnthropicStreamAccumulator::
                                is_terminal_sse_data(data)
                            {
                                stream_done = true;
                                break;
                            }
                        } else {
                            let chunk: ChatCompletionChunk = serde_json::from_str(data)
                                .map_err(|e| {
                                    fail_partial(ChatError::Serde(e), &messages, injected_prefix)
                                })?;
                            for choice in &chunk.choices {
                                if let Some(text) = choice.delta.reasoning_text() {
                                    on_reasoning_delta(text.clone());
                                    emit_reasoning_delta(
                                        options.status_emitter.as_ref(),
                                        &request_id,
                                        &options.model,
                                        api_calls,
                                        &text,
                                    )
                                    .await;
                                }
                            }
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
                                    dispatch_ctx.clone(),
                                );
                            }
                        }
                    }
                }
            }
        }

        if retry_stalled_round {
            stall_retries += 1;
            api_calls = api_calls.saturating_sub(1);
            tracing::warn!(
                request_id = %request_id,
                model = %options.model,
                attempt = stall_retries,
                "LLM stream stalled before first token — retrying round"
            );
            sleep(Duration::from_millis(400)).await;
            continue;
        }

        if !is_anthropic {
            for tc in tool_dispatch.drain_at_round_end() {
                push_early_stream_tool(
                    tc,
                    &mut in_flight,
                    &mut dispatched_tool_ids,
                    dispatch_ctx.clone(),
                );
            }
            while let Some(result) = in_flight.next().await {
                let (id, msg) = result.map_err(|e| fail_partial(e, &messages, injected_prefix))?;
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
                provider_blocks: None,
            }
        };
        finish_llm_round(
            &llm,
            &model_ref,
            &round.content,
            &round.tool_calls,
            round.finish_reason.as_deref(),
            round.usage.as_ref(),
        );

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
            ev.estimated_cost_usd = ev
                .usage
                .as_mut()
                .map(|u| apply_resolved_cost_usd(&model_used, u, u.cost_usd));
            ev
        })
        .await;

        if let Some(ref u) = round.usage {
            if u.prompt_tokens > 0 {
                last_prompt_tokens = Some(u.prompt_tokens);
            }
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
            .map_err(|e| fail_partial(ChatError::Hook(e), &messages, injected_prefix))?;

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
                injected_prefix,
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
            provider_blocks: round.provider_blocks.clone(),
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
                let catalog = (active_set.mode() == ToolMode::Code)
                    .then(|| active_set.route_catalog(&matched));
                let router_call_id = router_call_id_from_calls(&round.tool_calls);
                active_set.apply_route(matched);
                if active_set.mode() == ToolMode::Code {
                    stream_code_allowlist = Some(Arc::new(active_set.code_allowlist()));
                    dispatch_ctx.code_allowlist = stream_code_allowlist.clone();
                }
                let matched_names: Vec<String> = if active_set.mode() == ToolMode::Code {
                    active_set.code_allowlist().into_iter().collect()
                } else {
                    active_set
                        .specs_for_llm()
                        .unwrap_or(&[])
                        .iter()
                        .filter(|s| !s.static_tool && s.name != crate::tools::ROUTER_TOOL_NAME)
                        .map(|s| s.name.clone())
                        .collect()
                };
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
                if let Some(catalog) = catalog {
                    messages.push(msg.clone());
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(MessageContent::Text(
                            serde_json::to_string(&catalog).unwrap_or_else(|_| "{}".into()),
                        )),
                        tool_calls: None,
                        tool_call_id: router_call_id,
                        name: Some(crate::tools::ROUTER_TOOL_NAME.to_string()),
                        refusal: None,
                        provider_blocks: None,
                    });
                }
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
                        provider_blocks: None,
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
                let results = futures_util::future::join_all(
                    function_tcs
                        .iter()
                        .map(|tc| dispatch_one(tc, dispatch_ctx.clone())),
                )
                .await;
                for r in results {
                    messages.push(r.map_err(|e| fail_partial(e, &messages, injected_prefix))?);
                }
            }

            if let Some(store) = &options.image_store {
                let store = store.lock().await;
                let _ = crate::images::attach_vision_from_tool_results(&mut messages, &store);
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
                        injected_prefix,
                    ));
                }
            }
        }

        let client_messages = conversation_messages_for_client(&messages, injected_prefix, 0);
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
        crate::telemetry::openinference::set_output_text(&outcome.content);
        return Ok(outcome);
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::truncate_tool_result;
    use crate::openai::ChatMessage;

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
    fn conversation_messages_for_client_strips_all_injected_system_rows() {
        let messages = vec![
            ChatMessage::text("system", "stable base"),
            ChatMessage::text("system", "dynamic tail"),
            ChatMessage::text("user", "hello"),
            ChatMessage::text("assistant", "hi"),
        ];
        let client = super::conversation_messages_for_client(&messages, 2, 0);
        assert_eq!(client.len(), 2);
        assert_eq!(client[0].role, "user");
        assert_eq!(client[1].role, "assistant");
    }

    #[test]
    fn conversation_messages_for_client_keeps_caller_system_row() {
        let messages = vec![
            ChatMessage::text("system", "injected"),
            ChatMessage::text("system", "caller supplied"),
            ChatMessage::text("user", "hello"),
        ];
        let client = super::conversation_messages_for_client(&messages, 1, 0);
        assert_eq!(client.len(), 2);
        assert_eq!(client[0].role, "system");
        assert_eq!(
            client[0].content.as_ref().and_then(|c| c.as_text()),
            Some("caller supplied")
        );
    }

    #[test]
    fn conversation_messages_for_client_strips_volatile_suffix() {
        let messages = vec![
            ChatMessage::text("system", "base"),
            ChatMessage::text("user", "hello"),
            ChatMessage::text("system", "todos"),
        ];
        let client = super::conversation_messages_for_client(&messages, 1, 1);
        assert_eq!(client.len(), 1);
        assert_eq!(client[0].role, "user");
    }

    #[tokio::test]
    async fn output_block_confident_no_skips_regen() {
        use crate::http::{ClientConfig, HttpClient, RetryPolicy};
        use crate::providers::{ProviderCredentials, ProviderId};
        use serde_json::json;
        use std::sync::Arc;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "jev-latest",
                "answers": { "regen": { "type": "noul", "noul": 0.1 } },
                "usage": { "input_tokens": 4, "output_tokens": 1 }
            })))
            .mount(&server)
            .await;

        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::TypeSafe, "ts-test");
        creds.insert_base_url(ProviderId::TypeSafe, server.uri());
        let http = HttpClient::new(ClientConfig {
            retry: RetryPolicy {
                max_retries: 0,
                ..RetryPolicy::default()
            },
            ..ClientConfig::default()
        })
        .expect("http");
        let mut options = super::ChatOptions::default();
        options.provider_credentials = Some(Arc::new(creds));
        assert!(!super::output_block_should_regen(&http, &options, "hello world", "policy").await);
    }

    #[tokio::test]
    async fn output_block_timeout_regens() {
        use crate::http::{ClientConfig, HttpClient, RetryPolicy};
        use crate::providers::{ProviderCredentials, ProviderId};
        use serde_json::json;
        use std::sync::Arc;
        use std::time::Duration;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({
                        "model": "jev-latest",
                        "answers": { "regen": { "type": "noul", "noul": 0.1 } },
                        "usage": { "input_tokens": 4, "output_tokens": 1 }
                    }))
                    .set_delay(Duration::from_secs(2)),
            )
            .mount(&server)
            .await;

        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::TypeSafe, "ts-test");
        creds.insert_base_url(ProviderId::TypeSafe, server.uri());
        let http = HttpClient::new(ClientConfig {
            retry: RetryPolicy {
                max_retries: 0,
                ..RetryPolicy::default()
            },
            ..ClientConfig::default()
        })
        .expect("http");
        let mut options = super::ChatOptions::default();
        options.provider_credentials = Some(Arc::new(creds));
        assert!(super::output_block_should_regen(&http, &options, "hello world", "policy").await);
    }

    #[test]
    fn prepend_system_messages_counts_non_empty_blocks() {
        let mut options = super::ChatOptions::new("https://api.openai.com", "k", "m");
        options.system_prompt_blocks = Some(vec![
            super::SystemPromptBlock::cached("base"),
            super::SystemPromptBlock::uncached("dynamic"),
            super::SystemPromptBlock::uncached(""),
        ]);
        let mut messages = Vec::new();
        let injected = super::prepend_system_messages(&options, &mut messages);
        assert_eq!(injected, 2);
        assert_eq!(messages.len(), 2);
        assert!(super::has_system_prompt(&options));
    }

    #[test]
    fn volatile_suffix_is_sent_and_stripped_from_client_messages() {
        let mut options = super::ChatOptions::new("https://api.openai.com", "k", "m");
        options.system_prompt_blocks = Some(vec![super::SystemPromptBlock::cached("base")]);
        options.volatile_suffix_blocks =
            Some(vec![super::SystemPromptBlock::uncached("session todos")]);
        let mut working = Vec::new();
        let prefix = super::prepend_system_messages(&options, &mut working);
        working.push(ChatMessage::text("user", "hello"));
        let request = super::messages_with_volatile_suffix(&options, &working);
        assert_eq!(request.len(), 3);
        assert_eq!(request[0].role, "system");
        assert_eq!(request[1].role, "user");
        assert_eq!(request[2].role, "system");
        assert_eq!(
            request[2].content.as_ref().and_then(|c| c.as_text()),
            Some("session todos")
        );
        let suffix = super::injected_system_suffix_count(&options);
        let client = super::conversation_messages_for_client(&request, prefix, suffix);
        assert_eq!(client.len(), 1);
        assert_eq!(client[0].role, "user");
    }

    #[test]
    fn prompt_prefix_hash_ignores_volatile_suffix() {
        let mut stable = super::ChatOptions::new("https://api.openai.com", "k", "m");
        stable.system_prompt_blocks = Some(vec![super::SystemPromptBlock::cached("base")]);
        let mut with_tail = stable.clone();
        with_tail.volatile_suffix_blocks =
            Some(vec![super::SystemPromptBlock::uncached("todos v2")]);
        let tools = ["read", "write"];
        assert_eq!(
            super::prompt_prefix_hash(&stable, &tools),
            super::prompt_prefix_hash(&with_tail, &tools)
        );
        with_tail.system_prompt_blocks =
            Some(vec![super::SystemPromptBlock::cached("base changed")]);
        assert_ne!(
            super::prompt_prefix_hash(&stable, &tools),
            super::prompt_prefix_hash(&with_tail, &tools)
        );
    }

    #[test]
    fn fail_partial_wraps_cause_with_messages() {
        let messages = vec![
            ChatMessage::text("user", "hello"),
            ChatMessage::text("assistant", "partial"),
        ];
        let err = super::fail_partial(super::ChatError::MaxToolRounds(3), &messages, 0);
        let partial = err.partial_messages().expect("partial messages");
        assert_eq!(partial.len(), 2);
        assert!(matches!(
            err.root_cause(),
            super::ChatError::MaxToolRounds(3)
        ));
        assert_eq!(err.to_string(), "exceeded max tool rounds (3)");
    }

    #[test]
    fn fail_partial_keeps_messages_on_cancel() {
        let messages = vec![
            ChatMessage::text("user", "hello"),
            ChatMessage::text("assistant", "partial"),
        ];
        let err = super::fail_partial(super::ChatError::Cancelled, &messages, 0);
        assert!(err.is_cancelled());
        let partial = err.partial_messages().expect("partial messages");
        assert_eq!(partial.len(), 2);
        assert_eq!(err.to_string(), "request cancelled");
    }

    #[test]
    fn detects_transient_stream_stalls() {
        assert!(super::detail_is_transient_stream_stall(
            "HTTP request failed: error decoding response body"
        ));
        assert!(super::detail_is_transient_stream_stall(
            "operation timed out"
        ));
        assert!(super::detail_is_transient_stream_stall("connection reset"));
        assert!(!super::detail_is_transient_stream_stall(
            "HTTP 400: invalid_request"
        ));
    }

    #[test]
    fn retries_stalled_round_only_before_tokens() {
        let err = super::ChatError::Api("error decoding response body".into());
        assert!(super::should_retry_stalled_stream_round(&err, 0, "", false));
        assert!(super::should_retry_stalled_stream_round(&err, 1, "", false));
        assert!(!super::should_retry_stalled_stream_round(
            &err, 2, "", false
        ));
        assert!(!super::should_retry_stalled_stream_round(
            &err, 0, "hello", false
        ));
        assert!(!super::should_retry_stalled_stream_round(&err, 0, "", true));
        assert!(!super::should_retry_stalled_stream_round(
            &super::ChatError::Cancelled,
            0,
            "",
            false
        ));
    }
}

#[cfg(test)]
mod dispatch_freshness_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use async_trait::async_trait;
    use serde_json::{Value, json};
    use tokio::sync::Mutex;

    use super::{DispatchCtx, dispatch_one, new_tool_loop_guard};
    use crate::hooks::HookRegistry;
    use crate::openai::{FunctionCall, ToolCall};
    use crate::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

    struct CountingRead {
        calls: AtomicU32,
        content: Arc<Mutex<String>>,
    }

    #[async_trait]
    impl Tool for CountingRead {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("read", json!({"type": "object"}))
        }

        async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let content = self.content.lock().await.clone();
            Ok(json!({ "ok": true, "content": content }))
        }
    }

    struct CountingEdit {
        calls: AtomicU32,
        content: Arc<Mutex<String>>,
    }

    #[async_trait]
    impl Tool for CountingEdit {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("edit", json!({"type": "object"}))
        }

        async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let new_string = arguments
                .get("new_string")
                .and_then(Value::as_str)
                .unwrap_or("edited")
                .to_string();
            *self.content.lock().await = new_string.clone();
            Ok(json!({ "ok": true, "rel_path": "src/foo.rs" }))
        }
    }

    fn tool_call(id: &str, name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.to_string(),
            },
        }
    }

    #[tokio::test]
    async fn read_edit_read_invokes_read_twice_and_returns_post_edit_state() {
        let content = Arc::new(Mutex::new("before".to_string()));
        let read = Arc::new(CountingRead {
            calls: AtomicU32::new(0),
            content: Arc::clone(&content),
        });
        let edit = Arc::new(CountingEdit {
            calls: AtomicU32::new(0),
            content: Arc::clone(&content),
        });
        let registry = ToolRegistry::new();
        registry
            .register(read.clone())
            .await
            .expect("register read");
        registry
            .register(edit.clone())
            .await
            .expect("register edit");
        let hooks = HookRegistry::new();
        let loop_guard = new_tool_loop_guard();
        let ctx = DispatchCtx {
            hooks: &hooks,
            registry: &registry,
            status_emitter: None,
            request_id: "req",
            round: 1,
            model: "test",
            emit_start: false,
            tool_result_max_chars: 0,
            loop_guard: Some(&loop_guard),
            cancel: None,
            code_allowlist: None,
        };

        let first_read = dispatch_one(
            &tool_call("1", "read", json!({"path": "src/foo.rs"})),
            ctx.clone(),
        )
        .await
        .expect("first read");
        let first_text = first_read
            .content
            .as_ref()
            .and_then(|content| content.as_text())
            .expect("first read text");
        assert!(first_text.contains("before"));

        dispatch_one(
            &tool_call(
                "2",
                "edit",
                json!({
                    "path": "src/foo.rs",
                    "old_string": "before",
                    "new_string": "after"
                }),
            ),
            ctx.clone(),
        )
        .await
        .expect("edit");

        let second_read = dispatch_one(&tool_call("3", "read", json!({"path": "src/foo.rs"})), ctx)
            .await
            .expect("second read");
        let second_text = second_read
            .content
            .as_ref()
            .and_then(|content| content.as_text())
            .expect("second read text");
        assert!(
            second_text.contains("after"),
            "second read must return post-edit state, got {second_text}"
        );
        assert!(!second_text.contains("before"));
        assert_eq!(read.calls.load(Ordering::SeqCst), 2);
        assert_eq!(edit.calls.load(Ordering::SeqCst), 1);
    }
}
