//! OpenInference attributes on OpenTelemetry spans.
//!
//! OpenTelemetry moves the spans. [OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)
//! names what an LLM span means. This module writes those names when the `otlp`
//! feature is on.
//!
//! Payload text follows [`ScrubMode`] from [`super::init_tracing`]. The default
//! is [`ScrubMode::Redact`]. Roles, tool names, model ids, and token counts stay
//! visible. Message text, tool arguments, and tool results do not.
//!
//! `data:` URLs are replaced with `[data-uri]` in every mode. Each string
//! attribute is capped at 16_384 characters.

#[cfg(feature = "otlp")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::openai::ChatMessage;
#[cfg(any(test, feature = "otlp"))]
use crate::openai::{ContentPart, MessageContent};
use crate::proto;
use crate::tools::ToolSpec;

use super::scrub::ScrubMode;

/// Required on every OpenInference span.
pub const SPAN_KIND: &str = "openinference.span.kind";

const MAX_CHARS: usize = 16_384;

const MIME_TEXT: &str = "text/plain";
const MIME_JSON: &str = "application/json";

/// OpenInference `openinference.span.kind` values used by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpanKind {
    Llm,
    Chain,
    Tool,
    Agent,
    Guardrail,
}

impl SpanKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "LLM",
            Self::Chain => "CHAIN",
            Self::Tool => "TOOL",
            Self::Agent => "AGENT",
            Self::Guardrail => "GUARDRAIL",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AttrValue {
    Str(String),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Attr {
    pub key: String,
    pub value: AttrValue,
}

/// Sampling fields copied onto `llm.invocation_parameters`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct InvocationParams<'a> {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_completion_tokens: Option<u32>,
    pub reasoning_effort: Option<&'a str>,
    pub seed: Option<i64>,
}

/// One model call to record as an `LLM` span.
pub(crate) struct LlmStart<'a> {
    pub session_id: &'a str,
    pub model_name: &'a str,
    pub provider: &'a str,
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [ToolSpec]>,
    pub params: &'a InvocationParams<'a>,
}

/// Fields known after the provider returns.
#[cfg(any(test, feature = "otlp"))]
pub(crate) struct LlmFinish<'a> {
    pub model_name: &'a str,
    pub provider: &'a str,
    pub finish_reason: Option<&'a str>,
    pub usage: Option<&'a proto::Usage>,
    pub output: &'a ChatMessage,
}

static PAYLOAD_SCRUB: AtomicU8 = AtomicU8::new(0);

pub(crate) fn set_payload_scrub(mode: ScrubMode) {
    let code = match mode {
        ScrubMode::Redact => 0,
        ScrubMode::Hash => 1,
        ScrubMode::Allow => 2,
    };
    PAYLOAD_SCRUB.store(code, Ordering::Relaxed);
}

fn payload_mode() -> ScrubMode {
    match PAYLOAD_SCRUB.load(Ordering::Relaxed) {
        1 => ScrubMode::Hash,
        2 => ScrubMode::Allow,
        _ => ScrubMode::Redact,
    }
}

fn str_attr(key: impl Into<String>, value: impl Into<String>) -> Attr {
    Attr {
        key: key.into(),
        value: AttrValue::Str(value.into()),
    }
}

fn int_attr(key: impl Into<String>, value: i64) -> Attr {
    Attr {
        key: key.into(),
        value: AttrValue::Int(value),
    }
}

fn float_attr(key: impl Into<String>, value: f64) -> Attr {
    Attr {
        key: key.into(),
        value: AttrValue::Float(value),
    }
}

fn truncate(value: &str) -> String {
    if value.chars().count() <= MAX_CHARS {
        return value.to_string();
    }
    let mut out: String = value.chars().take(MAX_CHARS).collect();
    out.push('…');
    out
}

fn hash_text(value: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("[HASH:{:08x}]", hasher.finish() as u32)
}

fn sensitive(value: &str, mode: ScrubMode) -> String {
    match mode {
        ScrubMode::Redact => "[REDACTED]".to_string(),
        ScrubMode::Hash => hash_text(value),
        ScrubMode::Allow => truncate(value),
    }
}

#[cfg(any(test, feature = "otlp"))]
fn scrub_image_url(url: &str, mode: ScrubMode) -> String {
    if url.starts_with("data:") {
        return "[data-uri]".to_string();
    }
    sensitive(url, mode)
}

#[cfg(any(test, feature = "otlp"))]
pub(crate) fn invocation_json(params: &InvocationParams<'_>) -> String {
    let mut map = serde_json::Map::new();
    if let Some(value) = params.temperature {
        map.insert("temperature".into(), serde_json::json!(value));
    }
    if let Some(value) = params.top_p {
        map.insert("top_p".into(), serde_json::json!(value));
    }
    if let Some(value) = params.max_completion_tokens {
        map.insert("max_tokens".into(), serde_json::json!(value));
    }
    if let Some(value) = params.reasoning_effort {
        map.insert("reasoning_effort".into(), serde_json::json!(value));
    }
    if let Some(value) = params.seed {
        map.insert("seed".into(), serde_json::json!(value));
    }
    serde_json::Value::Object(map).to_string()
}

fn push_session(attrs: &mut Vec<Attr>, session_id: &str) {
    if !session_id.is_empty() {
        attrs.push(str_attr("session.id", session_id));
    }
}

fn push_usage(attrs: &mut Vec<Attr>, usage: &proto::Usage) {
    attrs.push(int_attr(
        "llm.token_count.prompt",
        i64::from(usage.prompt_tokens),
    ));
    attrs.push(int_attr(
        "llm.token_count.completion",
        i64::from(usage.completion_tokens),
    ));
    attrs.push(int_attr(
        "llm.token_count.total",
        i64::from(usage.total_tokens),
    ));
    if let Some(cached) = usage.cached_tokens {
        attrs.push(int_attr(
            "llm.token_count.prompt_details.cache_read",
            i64::from(cached),
        ));
    }
    if let Some(reasoning) = usage.reasoning_tokens {
        attrs.push(int_attr(
            "llm.token_count.completion_details.reasoning",
            i64::from(reasoning),
        ));
    }
    if let Some(cost) = usage.cost_usd {
        attrs.push(float_attr("llm.cost.total", cost));
    }
}

#[cfg(any(test, feature = "otlp"))]
fn push_message(
    attrs: &mut Vec<Attr>,
    prefix: &str,
    index: usize,
    msg: &ChatMessage,
    mode: ScrubMode,
) {
    let base = format!("{prefix}.{index}.message");
    attrs.push(str_attr(format!("{base}.role"), msg.role.clone()));
    if let Some(name) = &msg.name {
        attrs.push(str_attr(format!("{base}.name"), name.clone()));
    }
    if let Some(id) = &msg.tool_call_id {
        attrs.push(str_attr(format!("{base}.tool_call_id"), id.clone()));
    }
    match msg.content.as_ref() {
        Some(MessageContent::Text(text)) => {
            attrs.push(str_attr(format!("{base}.content"), sensitive(text, mode)));
        }
        Some(MessageContent::Parts(parts)) => push_parts(attrs, &base, parts, mode),
        None => {
            if let Some(refusal) = &msg.refusal {
                attrs.push(str_attr(
                    format!("{base}.content"),
                    sensitive(refusal, mode),
                ));
            }
        }
    }
    if let Some(calls) = &msg.tool_calls {
        for (call_index, call) in calls.iter().enumerate() {
            let call_base = format!("{base}.tool_calls.{call_index}.tool_call");
            if !call.id.is_empty() {
                attrs.push(str_attr(format!("{call_base}.id"), call.id.clone()));
            }
            attrs.push(str_attr(
                format!("{call_base}.function.name"),
                call.function.name.clone(),
            ));
            attrs.push(str_attr(
                format!("{call_base}.function.arguments"),
                sensitive(&call.function.arguments, mode),
            ));
        }
    }
}

#[cfg(any(test, feature = "otlp"))]
fn push_parts(attrs: &mut Vec<Attr>, base: &str, parts: &[ContentPart], mode: ScrubMode) {
    for (index, part) in parts.iter().enumerate() {
        let part_base = format!("{base}.contents.{index}.message_content");
        match part {
            ContentPart::Text { text } => {
                attrs.push(str_attr(format!("{part_base}.type"), "text"));
                attrs.push(str_attr(format!("{part_base}.text"), sensitive(text, mode)));
            }
            ContentPart::ImageUrl { image_url } => {
                attrs.push(str_attr(format!("{part_base}.type"), "image"));
                attrs.push(str_attr(
                    format!("{part_base}.image.image.url"),
                    scrub_image_url(&image_url.url, mode),
                ));
            }
            ContentPart::ImageRef { hash, .. } => {
                attrs.push(str_attr(format!("{part_base}.type"), "image"));
                attrs.push(str_attr(
                    format!("{part_base}.image.image.url"),
                    sensitive(&format!("image-ref:{hash}"), mode),
                ));
            }
            ContentPart::InputAudio { .. } => {
                attrs.push(str_attr(format!("{part_base}.type"), "audio"));
                attrs.push(str_attr(format!("{part_base}.audio.audio.url"), "[audio]"));
            }
            ContentPart::File { file } => {
                attrs.push(str_attr(format!("{part_base}.type"), "text"));
                let label = file
                    .filename
                    .as_deref()
                    .or(file.file_id.as_deref())
                    .unwrap_or("file");
                attrs.push(str_attr(
                    format!("{part_base}.text"),
                    sensitive(label, mode),
                ));
            }
        }
    }
}

#[cfg(any(test, feature = "otlp"))]
fn push_tools(attrs: &mut Vec<Attr>, tools: &[ToolSpec]) {
    for (index, spec) in tools.iter().enumerate() {
        let base = format!("llm.tools.{index}.tool");
        attrs.push(str_attr(format!("{base}.name"), spec.name.clone()));
        if let Some(description) = &spec.description {
            attrs.push(str_attr(format!("{base}.description"), description.clone()));
        }
        let mut function = serde_json::Map::new();
        function.insert("name".into(), serde_json::json!(spec.name));
        if let Some(description) = &spec.description {
            function.insert("description".into(), serde_json::json!(description));
        }
        function.insert("parameters".into(), spec.parameters_schema.clone());
        let schema = serde_json::json!({
            "type": "function",
            "function": function,
        });
        attrs.push(str_attr(
            format!("{base}.json_schema"),
            truncate(&schema.to_string()),
        ));
    }
}

#[cfg(any(test, feature = "otlp"))]
pub(crate) fn llm_request_attributes(start: &LlmStart<'_>, mode: ScrubMode) -> Vec<Attr> {
    let mut attrs = Vec::new();
    attrs.push(str_attr(SPAN_KIND, SpanKind::Llm.as_str()));
    push_session(&mut attrs, start.session_id);
    if !start.model_name.is_empty() {
        attrs.push(str_attr("llm.model_name", start.model_name));
        attrs.push(str_attr("llm.request.model_name", start.model_name));
    }
    if !start.provider.is_empty() {
        attrs.push(str_attr("llm.provider", start.provider));
        attrs.push(str_attr("llm.system", start.provider));
    }
    attrs.push(str_attr(
        "llm.invocation_parameters",
        invocation_json(start.params),
    ));
    for (index, message) in start.messages.iter().enumerate() {
        push_message(&mut attrs, "llm.input_messages", index, message, mode);
    }
    if let Some(tools) = start.tools {
        push_tools(&mut attrs, tools);
    }
    attrs
}

#[cfg(any(test, feature = "otlp"))]
pub(crate) fn llm_response_attributes(done: &LlmFinish<'_>, mode: ScrubMode) -> Vec<Attr> {
    let mut attrs = Vec::new();
    if !done.model_name.is_empty() {
        attrs.push(str_attr("llm.model_name", done.model_name));
        attrs.push(str_attr("llm.response.model_name", done.model_name));
    }
    if !done.provider.is_empty() {
        attrs.push(str_attr("llm.provider", done.provider));
        attrs.push(str_attr("llm.system", done.provider));
    }
    if let Some(reason) = done.finish_reason {
        attrs.push(str_attr("llm.finish_reason", reason));
    }
    if let Some(usage) = done.usage {
        push_usage(&mut attrs, usage);
    }
    push_message(&mut attrs, "llm.output_messages", 0, done.output, mode);
    attrs
}

pub(crate) fn chain_attributes(session_id: &str) -> Vec<Attr> {
    let mut attrs = vec![str_attr(SPAN_KIND, SpanKind::Chain.as_str())];
    push_session(&mut attrs, session_id);
    attrs
}

pub(crate) fn agent_attributes(agent_id: &str, session_id: &str) -> Vec<Attr> {
    let mut attrs = vec![str_attr(SPAN_KIND, SpanKind::Agent.as_str())];
    push_session(&mut attrs, session_id);
    if !agent_id.is_empty() {
        attrs.push(str_attr("agent.name", agent_id));
        attrs.push(str_attr("agent.id", agent_id));
    }
    attrs
}

pub(crate) fn tool_attributes(
    session_id: &str,
    name: &str,
    tool_id: &str,
    arguments: &str,
    mode: ScrubMode,
) -> Vec<Attr> {
    let mut attrs = vec![str_attr(SPAN_KIND, SpanKind::Tool.as_str())];
    push_session(&mut attrs, session_id);
    attrs.push(str_attr("tool.name", name));
    if !tool_id.is_empty() {
        attrs.push(str_attr("tool.id", tool_id));
    }
    attrs.push(str_attr("input.mime_type", MIME_JSON));
    attrs.push(str_attr("input.value", sensitive(arguments, mode)));
    attrs
}

pub(crate) fn io_attributes(
    input: Option<&str>,
    output: Option<&str>,
    mime: &str,
    mode: ScrubMode,
) -> Vec<Attr> {
    let mut attrs = Vec::new();
    if let Some(value) = input {
        attrs.push(str_attr("input.mime_type", mime));
        attrs.push(str_attr("input.value", sensitive(value, mode)));
    }
    if let Some(value) = output {
        attrs.push(str_attr("output.mime_type", mime));
        attrs.push(str_attr("output.value", sensitive(value, mode)));
    }
    attrs
}

pub(crate) fn guardrail_attributes(stage: &str) -> Vec<Attr> {
    vec![
        str_attr(SPAN_KIND, SpanKind::Guardrail.as_str()),
        str_attr("metadata", format!(r#"{{"stage":"{stage}"}}"#)),
    ]
}

pub(crate) fn usage_attributes(usage: &proto::Usage) -> Vec<Attr> {
    let mut attrs = Vec::new();
    push_usage(&mut attrs, usage);
    attrs
}

#[cfg(feature = "otlp")]
struct ActiveSpan {
    span: tracing::Span,
    finished: AtomicBool,
}

/// Child `ChatCompletion` span for one model call.
///
/// The span starts when [`LlmSpan::begin_chat`] runs and ends when this value
/// drops. Call `finish` after a response, or [`LlmSpan::fail`] on an error.
/// A drop without either call marks the span as an error when `otlp` is on.
pub(crate) struct LlmSpan {
    #[cfg(feature = "otlp")]
    inner: Option<ActiveSpan>,
}

impl LlmSpan {
    pub(crate) fn begin_chat(start: &LlmStart<'_>) -> Self {
        #[cfg(feature = "otlp")]
        {
            let span = tracing::info_span!("ChatCompletion");
            apply(&span, &llm_request_attributes(start, payload_mode()));
            Self {
                inner: Some(ActiveSpan {
                    span,
                    finished: AtomicBool::new(false),
                }),
            }
        }
        #[cfg(not(feature = "otlp"))]
        {
            let _ = (
                start.session_id,
                start.model_name,
                start.provider,
                start.messages.len(),
                start.tools.map(|tools| tools.len()),
                start.params.temperature,
                start.params.top_p,
                start.params.max_completion_tokens,
                start.params.reasoning_effort,
                start.params.seed,
            );
            Self {}
        }
    }

    pub(crate) fn fail(&self, message: &str) {
        #[cfg(feature = "otlp")]
        if let Some(inner) = &self.inner {
            mark_error(&inner.span, message);
            inner.finished.store(true, Ordering::Relaxed);
        }
        #[cfg(not(feature = "otlp"))]
        {
            let _ = (self, message);
        }
    }

    #[cfg(feature = "otlp")]
    pub(crate) fn finish(&self, done: &LlmFinish<'_>) {
        if let Some(inner) = &self.inner {
            apply(&inner.span, &llm_response_attributes(done, payload_mode()));
            inner.finished.store(true, Ordering::Relaxed);
        }
    }
}

impl Drop for LlmSpan {
    fn drop(&mut self) {
        #[cfg(feature = "otlp")]
        if let Some(inner) = &self.inner
            && !inner.finished.load(Ordering::Relaxed)
        {
            mark_error(&inner.span, "llm call ended without a recorded response");
        }
    }
}

/// `GUARDRAIL` span held for the duration of one guardrail chain.
pub(crate) struct GuardSpan {
    #[cfg(feature = "otlp")]
    span: Option<tracing::Span>,
}

impl GuardSpan {
    pub(crate) fn begin(stage: &str, input: &str) -> Self {
        #[cfg(feature = "otlp")]
        {
            let span = tracing::info_span!("guardrail");
            let mut attrs = guardrail_attributes(stage);
            attrs.extend(io_attributes(Some(input), None, MIME_TEXT, payload_mode()));
            apply(&span, &attrs);
            Self { span: Some(span) }
        }
        #[cfg(not(feature = "otlp"))]
        {
            let _ = (guardrail_attributes(stage), input);
            Self {}
        }
    }

    pub(crate) fn finish(&self, output: &str, blocked: bool) {
        #[cfg(feature = "otlp")]
        if let Some(span) = &self.span {
            apply(
                span,
                &io_attributes(None, Some(output), MIME_TEXT, payload_mode()),
            );
            if blocked {
                mark_error(span, "guardrail blocked content");
            }
        }
        #[cfg(not(feature = "otlp"))]
        {
            let _ = (self, output, blocked);
        }
    }
}

pub(crate) fn tag_chain(session_id: &str) {
    apply_current(&chain_attributes(session_id));
}

pub(crate) fn tag_agent(agent_id: &str, session_id: &str) {
    apply_current(&agent_attributes(agent_id, session_id));
}

pub(crate) fn tag_llm(session_id: &str, model_name: &str, provider: &str) {
    let mut attrs = vec![str_attr(SPAN_KIND, SpanKind::Llm.as_str())];
    push_session(&mut attrs, session_id);
    if !model_name.is_empty() {
        attrs.push(str_attr("llm.model_name", model_name));
    }
    if !provider.is_empty() {
        attrs.push(str_attr("llm.provider", provider));
        attrs.push(str_attr("llm.system", provider));
    }
    apply_current(&attrs);
}

pub(crate) fn tag_tool(session_id: &str, name: &str, tool_id: &str, arguments: &str) {
    apply_current(&tool_attributes(
        session_id,
        name,
        tool_id,
        arguments,
        payload_mode(),
    ));
}

pub(crate) fn set_input_text(value: &str) {
    apply_current(&io_attributes(Some(value), None, MIME_TEXT, payload_mode()));
}

pub(crate) fn set_output_text(value: &str) {
    apply_current(&io_attributes(None, Some(value), MIME_TEXT, payload_mode()));
}

pub(crate) fn set_output_json(value: &str) {
    apply_current(&io_attributes(None, Some(value), MIME_JSON, payload_mode()));
}

pub(crate) fn set_usage(usage: &proto::Usage) {
    apply_current(&usage_attributes(usage));
}

pub(crate) fn fail_current(message: &str) {
    #[cfg(feature = "otlp")]
    mark_error(&tracing::Span::current(), message);
    #[cfg(not(feature = "otlp"))]
    {
        let _ = message;
    }
}

fn apply_current(attrs: &[Attr]) {
    #[cfg(feature = "otlp")]
    apply(&tracing::Span::current(), attrs);
    #[cfg(not(feature = "otlp"))]
    {
        let _ = attrs;
    }
}

#[cfg(feature = "otlp")]
fn apply(span: &tracing::Span, attrs: &[Attr]) {
    use opentelemetry::Value;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    for attr in attrs {
        match &attr.value {
            AttrValue::Str(value) => {
                span.set_attribute(attr.key.clone(), Value::from(value.clone()));
            }
            AttrValue::Int(value) => span.set_attribute(attr.key.clone(), *value),
            AttrValue::Float(value) => span.set_attribute(attr.key.clone(), *value),
        }
    }
}

#[cfg(feature = "otlp")]
fn mark_error(span: &tracing::Span, message: &str) {
    use opentelemetry::trace::Status;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    span.set_status(Status::error(truncate(message)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{ChatMessage, FunctionCall, ToolCall};
    use crate::tools::ToolSpec;

    fn attr<'a>(attrs: &'a [Attr], key: &str) -> &'a AttrValue {
        &attrs
            .iter()
            .find(|attr| attr.key == key)
            .unwrap_or_else(|| panic!("missing {key}"))
            .value
    }

    fn text_message(role: &str, content: &str) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    #[test]
    fn llm_request_uses_openinference_keys_and_redacts_content() {
        let messages = vec![
            text_message("system", "You are a support agent."),
            text_message("user", "Where is order 1842?"),
        ];
        let tools = vec![
            ToolSpec::new("lookup_order", serde_json::json!({"type": "object"}))
                .with_description("Look up an order"),
        ];
        let params = InvocationParams {
            temperature: Some(0.2),
            max_completion_tokens: Some(512),
            ..InvocationParams::default()
        };
        let attrs = llm_request_attributes(
            &LlmStart {
                session_id: "session-7f3a",
                model_name: "gpt-4o-mini",
                provider: "openai",
                messages: &messages,
                tools: Some(&tools),
                params: &params,
            },
            ScrubMode::Redact,
        );

        assert_eq!(attr(&attrs, SPAN_KIND), &AttrValue::Str("LLM".into()));
        assert_eq!(
            attr(&attrs, "llm.model_name"),
            &AttrValue::Str("gpt-4o-mini".into())
        );
        assert_eq!(
            attr(&attrs, "llm.provider"),
            &AttrValue::Str("openai".into())
        );
        assert_eq!(attr(&attrs, "llm.system"), &AttrValue::Str("openai".into()));
        assert_eq!(
            attr(&attrs, "session.id"),
            &AttrValue::Str("session-7f3a".into())
        );
        assert_eq!(
            attr(&attrs, "llm.input_messages.0.message.role"),
            &AttrValue::Str("system".into())
        );
        assert_eq!(
            attr(&attrs, "llm.input_messages.0.message.content"),
            &AttrValue::Str("[REDACTED]".into())
        );
        assert_eq!(
            attr(&attrs, "llm.input_messages.1.message.role"),
            &AttrValue::Str("user".into())
        );
        assert_eq!(
            attr(&attrs, "llm.tools.0.tool.name"),
            &AttrValue::Str("lookup_order".into())
        );
        let parameters = attr(&attrs, "llm.invocation_parameters");
        let AttrValue::Str(json) = parameters else {
            panic!("invocation parameters must be a string");
        };
        assert!(json.contains("\"temperature\":0.2"));
        assert!(json.contains("\"max_tokens\":512"));
    }

    #[test]
    fn llm_response_records_tool_call_and_token_counts() {
        let output = ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "lookup_order".into(),
                    arguments: r#"{"order_id":"1842"}"#.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
            refusal: None,
            provider_blocks: None,
        };
        let usage = proto::Usage {
            prompt_tokens: 212,
            completion_tokens: 18,
            total_tokens: 230,
            cached_tokens: Some(5),
            reasoning_tokens: None,
            cost_usd: Some(0.001),
        };
        let attrs = llm_response_attributes(
            &LlmFinish {
                model_name: "gpt-4o-mini",
                provider: "openai",
                finish_reason: Some("tool_calls"),
                usage: Some(&usage),
                output: &output,
            },
            ScrubMode::Allow,
        );

        assert_eq!(
            attr(&attrs, "llm.output_messages.0.message.role"),
            &AttrValue::Str("assistant".into())
        );
        assert_eq!(
            attr(
                &attrs,
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
            ),
            &AttrValue::Str("lookup_order".into())
        );
        assert_eq!(
            attr(
                &attrs,
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
            ),
            &AttrValue::Str(r#"{"order_id":"1842"}"#.into())
        );
        assert_eq!(attr(&attrs, "llm.token_count.prompt"), &AttrValue::Int(212));
        assert_eq!(
            attr(&attrs, "llm.token_count.completion"),
            &AttrValue::Int(18)
        );
        assert_eq!(attr(&attrs, "llm.token_count.total"), &AttrValue::Int(230));
        assert_eq!(
            attr(&attrs, "llm.token_count.prompt_details.cache_read"),
            &AttrValue::Int(5)
        );
        assert_eq!(
            attr(&attrs, "llm.finish_reason"),
            &AttrValue::Str("tool_calls".into())
        );
        assert_eq!(attr(&attrs, "llm.cost.total"), &AttrValue::Float(0.001));
    }

    #[test]
    fn allow_mode_keeps_text_and_strips_data_uris() {
        let message = ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: crate::openai::ImageUrl {
                    url: "data:image/png;base64,AAAA".into(),
                    detail: None,
                },
            }])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
            provider_blocks: None,
        };
        let attrs = llm_request_attributes(
            &LlmStart {
                session_id: "",
                model_name: "m",
                provider: "openai",
                messages: &[message],
                tools: None,
                params: &InvocationParams::default(),
            },
            ScrubMode::Allow,
        );
        assert_eq!(
            attr(
                &attrs,
                "llm.input_messages.0.message.contents.0.message_content.image.image.url"
            ),
            &AttrValue::Str("[data-uri]".into())
        );
        assert!(attrs.iter().all(|attr| attr.key != "session.id"));
    }

    #[test]
    fn guardrail_span_kind_names_the_stage() {
        let attrs = guardrail_attributes("input");
        assert_eq!(attr(&attrs, SPAN_KIND), &AttrValue::Str("GUARDRAIL".into()));
        assert_eq!(
            attr(&attrs, "metadata"),
            &AttrValue::Str(r#"{"stage":"input"}"#.into())
        );
    }

    #[test]
    fn agent_and_tool_kinds_are_stable() {
        let agent = agent_attributes("support-triage-agent", "session-1");
        assert_eq!(attr(&agent, SPAN_KIND), &AttrValue::Str("AGENT".into()));
        assert_eq!(
            attr(&agent, "agent.id"),
            &AttrValue::Str("support-triage-agent".into())
        );
        assert_eq!(
            attr(&agent, "agent.name"),
            &AttrValue::Str("support-triage-agent".into())
        );

        let tool = tool_attributes(
            "session-1",
            "lookup_order",
            "call_1",
            "{\"a\":1}",
            ScrubMode::Hash,
        );
        assert_eq!(attr(&tool, SPAN_KIND), &AttrValue::Str("TOOL".into()));
        assert_eq!(
            attr(&tool, "tool.name"),
            &AttrValue::Str("lookup_order".into())
        );
        let AttrValue::Str(input) = attr(&tool, "input.value") else {
            panic!("input.value");
        };
        assert!(input.starts_with("[HASH:"));
    }
}
