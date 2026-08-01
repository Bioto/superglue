//! Async context optimization helpers (LLM calls live here to avoid module cycles).

use tracing::warn;

use crate::context::SummarizeContextConfig;
use crate::context::{
    AaakCompressor, COMPRESS_SYSTEM, COMPRESS_USER_PREFIX, condense_tool_round as condense_round,
};
use crate::context::{
    message_text, passthrough_aaak_context, passthrough_at_messages, transcript_from_messages,
};
use crate::http::HttpClient;
use crate::openai::{ChatMessage, MessageContent};
use crate::providers::ProviderCredentials;
use crate::tools::ToolSpec;

use super::{ChatError, ChatOptions, ContextBlockProvider, ContextEventLogger, provider_chat_post};

/// Use a fast LLM to select relevant tools; falls back to all dynamic tools on error.
pub async fn resolve_tool_route(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    user_context: &str,
    dynamic_specs: &[ToolSpec],
    route_model: &str,
    options: &ChatOptions,
    request_id: &str,
) -> Vec<ToolSpec> {
    if dynamic_specs.is_empty() {
        return Vec::new();
    }

    let mut tool_descriptions = String::new();
    for spec in dynamic_specs {
        let desc = spec
            .description
            .as_deref()
            .unwrap_or(spec.name.as_str())
            .lines()
            .next()
            .unwrap_or(spec.name.as_str());
        tool_descriptions.push_str(&format!("- {}: {desc}\n", spec.name));
    }

    let system = "You are a tool router. Given a user request, select which tools are needed. \
        Respond with a JSON array of tool names: [\"name1\", \"name2\", ...] \
        Use only the exact tool names from the list. If no tools are needed, use [].";
    let user_msg = format!(
        "User request:\n{user_context}\n\nAvailable tools:\n{tool_descriptions}\nWhich tools are needed? Respond with a JSON array only."
    );

    let messages = vec![
        ChatMessage::text("system", system),
        ChatMessage::text("user", user_msg),
    ];

    let mut route_options = options.clone();
    route_options.model = route_model.to_string();
    route_options.max_completion_tokens = Some(256);

    let response = match provider_chat_post(
        http,
        credentials,
        &messages,
        None,
        None,
        &route_options,
        request_id,
        0,
        false,
    )
    .await
    {
        Ok((val, _)) => val,
        Err(e) => {
            warn!(
                error = %e,
                count = dynamic_specs.len(),
                "tool routing failed; falling back to all dynamic tools"
            );
            return dynamic_specs.to_vec();
        }
    };

    let provider =
        crate::providers::resolve_provider(&crate::providers::parse_model_ref(route_model));
    let normalized = match provider.parse_chat_response(&response) {
        Ok(n) => n,
        Err(e) => {
            warn!(error = %e, "tool routing parse failed; falling back to all dynamic tools");
            return dynamic_specs.to_vec();
        }
    };

    let text = normalized.content.unwrap_or_default();
    let parsed: serde_json::Value = match serde_json::from_str(text.trim()) {
        Ok(v) => v,
        Err(e) => {
            warn!(
                error = %e,
                preview = %text.chars().take(200).collect::<String>(),
                "tool routing returned invalid JSON; falling back to all dynamic tools"
            );
            return dynamic_specs.to_vec();
        }
    };

    let Some(names) = parsed.as_array() else {
        warn!("tool routing returned non-list; falling back to all dynamic tools");
        return dynamic_specs.to_vec();
    };

    let name_to_spec: std::collections::HashMap<&str, &ToolSpec> =
        dynamic_specs.iter().map(|s| (s.name.as_str(), s)).collect();

    let mut result = Vec::new();
    for name_val in names {
        let Some(name) = name_val.as_str() else {
            continue;
        };
        if let Some(spec) = name_to_spec.get(name) {
            result.push((*spec).clone());
        } else {
            warn!(name, "tool routing returned unrecognized tool name");
        }
    }

    result
}

#[must_use]
pub fn user_context_for_route(messages: &[ChatMessage], router_query: &str) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| router_query.to_string())
}

const SUMMARIZE_SYSTEM: &str = "You are a conversation summarizer. Produce a concise, factual summary of the \
conversation history provided. Preserve all key facts, decisions, and context \
that would be needed to continue the conversation coherently. Do not add \
opinions or filler — just the essential content, in plain prose.";

const SUMMARIZE_USER_PREFIX: &str = "Summarize the following conversation history concisely so it can be used as \
context for continuing the conversation:\n\n";

/// Options for auxiliary LLM calls (summarize / compress) that must not inherit
/// sampling parameters the parent model rejects (e.g. reasoning models + temperature).
fn auxiliary_chat_options(base: &ChatOptions, model: &str) -> ChatOptions {
    let mut opts = base.clone();
    opts.model = model.to_string();
    opts.temperature = None;
    opts.top_p = None;
    opts.presence_penalty = None;
    opts.frequency_penalty = None;
    opts.reasoning_effort = None;
    opts
}

/// Approximate the serialized context size without depending on a tokenizer.
pub(crate) fn estimate_context_chars(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| {
            let content_chars = message
                .content
                .as_ref()
                .and_then(MessageContent::as_text)
                .map_or(0, |text| text.chars().count());
            let tool_call_chars = message.tool_calls.as_ref().map_or(0, |calls| {
                calls
                    .iter()
                    .map(|call| {
                        call.function.name.chars().count()
                            + call.function.arguments.chars().count()
                            + call.id.chars().count()
                    })
                    .sum()
            });
            message.role.chars().count() + content_chars + tool_call_chars + 16
        })
        .sum()
}

fn should_summarize(messages: &[ChatMessage], config: &SummarizeContextConfig) -> bool {
    if !config.enabled || messages.len() <= config.threshold {
        return false;
    }
    if config.max_chars > 0 && estimate_context_chars(messages) <= config.max_chars {
        return false;
    }
    let has_system = messages
        .first()
        .is_some_and(|message| message.role == "system");
    let start = usize::from(has_system);
    messages.len().saturating_sub(start) > config.keep_recent
}

/// Compress old messages when over the configured context-size budget.
pub async fn maybe_summarize_messages(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    messages: &mut Vec<ChatMessage>,
    config: &SummarizeContextConfig,
    aaak_enabled: bool,
    aaak_model: Option<&str>,
    summarize_model: &str,
    options: &ChatOptions,
    request_id: &str,
) -> Result<(), ChatError> {
    if !should_summarize(messages, config) {
        return Ok(());
    }
    let chars_before = estimate_context_chars(messages);

    let system_msg = messages.first().cloned();
    let has_system = system_msg.as_ref().is_some_and(|m| m.role == "system");
    let start = if has_system { 1 } else { 0 };
    if messages.len().saturating_sub(start) <= config.keep_recent {
        return Ok(());
    }

    let end = messages.len().saturating_sub(config.keep_recent);
    let old_messages: Vec<ChatMessage> = messages[start..end].to_vec();
    let tail: Vec<ChatMessage> = messages[end..].to_vec();

    let summary_content = if aaak_enabled {
        let compress_model = aaak_model.unwrap_or(summarize_model);
        match compress_messages_aaak(
            http,
            credentials,
            &old_messages,
            compress_model,
            options,
            request_id,
        )
        .await
        {
            Ok(encoded) if !encoded.trim().is_empty() => {
                format!("[AAAK CTX]\n{encoded}\n[/AAAK CTX]")
            }
            Ok(_) => return Ok(()),
            Err(e) => {
                warn!(error = %e, "AAAK context compression failed; continuing with original messages");
                return Ok(());
            }
        }
    } else {
        let transcript = old_messages
            .iter()
            .map(|m| {
                let role = m.role.to_uppercase();
                let text = m.content.as_ref().and_then(|c| c.as_text()).unwrap_or("");
                format!("{role}: {text}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let api_messages = vec![
            ChatMessage::text("system", SUMMARIZE_SYSTEM),
            ChatMessage::text("user", format!("{SUMMARIZE_USER_PREFIX}{transcript}")),
        ];

        let sum_options = auxiliary_chat_options(options, summarize_model);

        let (val, _) = provider_chat_post(
            http,
            credentials,
            &api_messages,
            None,
            None,
            &sum_options,
            request_id,
            0,
            false,
        )
        .await?;

        let provider =
            crate::providers::resolve_provider(&crate::providers::parse_model_ref(summarize_model));
        let normalized = provider
            .parse_chat_response(&val)
            .map_err(|e| ChatError::Http(crate::http::Error::InvalidJson(e.to_string())))?;
        let text = normalized.content.unwrap_or_default();
        if text.trim().is_empty() {
            return Ok(());
        }
        format!("[Conversation Summary]\n{text}")
    };

    let mut new_messages = Vec::new();
    if has_system {
        let mut sys = system_msg.unwrap();
        if aaak_enabled {
            AaakCompressor::ensure_preamble_in_system(&mut sys);
        }
        new_messages.push(sys);
    }
    new_messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text(summary_content)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
    });
    new_messages.extend(tail);
    let chars_after = estimate_context_chars(&new_messages);
    if let Some(logger) = &options.context_event_logger {
        logger.log(format!(
            "context summarize before_messages={} after_messages={} before_chars={} after_chars={} threshold={} max_chars={} keep_recent={}",
            messages.len(),
            new_messages.len(),
            chars_before,
            chars_after,
            config.threshold,
            config.max_chars,
            config.keep_recent,
        ));
    }
    *messages = new_messages;
    Ok(())
}

async fn compress_messages_aaak(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    old_messages: &[ChatMessage],
    compress_model: &str,
    options: &ChatOptions,
    request_id: &str,
) -> Result<String, ChatError> {
    if old_messages
        .iter()
        .any(|m| message_text(m).contains("[AAAK CTX]"))
    {
        return Ok(passthrough_aaak_context(old_messages));
    }

    if old_messages
        .iter()
        .any(|m| message_text(m).trim().starts_with("[AT]"))
    {
        return Ok(passthrough_at_messages(old_messages));
    }

    let transcript = transcript_from_messages(old_messages);
    let messages = vec![
        ChatMessage::text("system", COMPRESS_SYSTEM),
        ChatMessage::text("user", format!("{COMPRESS_USER_PREFIX}{transcript}")),
    ];

    let mut compress_options = auxiliary_chat_options(options, compress_model);
    compress_options.max_completion_tokens = Some(512);

    let (val, _) = provider_chat_post(
        http,
        credentials,
        &messages,
        None,
        None,
        &compress_options,
        request_id,
        0,
        false,
    )
    .await?;

    let provider =
        crate::providers::resolve_provider(&crate::providers::parse_model_ref(compress_model));
    let normalized = provider
        .parse_chat_response(&val)
        .map_err(|e| ChatError::Http(crate::http::Error::InvalidJson(e.to_string())))?;

    Ok(normalized.content.unwrap_or_default().trim().to_string())
}

pub fn condense_tool_round(
    messages: &mut Vec<ChatMessage>,
    aaak_tool_condensing: bool,
    context_block_provider: Option<&ContextBlockProvider>,
    context_event_logger: Option<&ContextEventLogger>,
) {
    let len_before = messages.len();
    let chars_before = estimate_context_chars(messages);
    condense_round(messages, aaak_tool_condensing);
    if messages.len() != len_before
        && let Some(logger) = context_event_logger
    {
        logger.log(format!(
            "context condense before_messages={} after_messages={} before_chars={} after_chars={} aaak={}",
            len_before,
            messages.len(),
            chars_before,
            estimate_context_chars(messages),
            aaak_tool_condensing,
        ));
    }
    let Some(provider) = context_block_provider else {
        return;
    };
    let context_block = provider.render();
    if context_block.trim().is_empty() {
        return;
    }
    if let Some(message) = messages.last_mut()
        && message.role == "user"
        && let Some(MessageContent::Text(content)) = message.content.as_mut()
    {
        content.push_str("\n\n");
        content.push_str(&context_block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{FunctionCall, ToolCall};

    fn tool_round() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "grep".into(),
                        arguments: r#"{"pattern":"needle"}"#.into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
                refusal: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(r#"{"count":1}"#.into())),
                tool_calls: None,
                tool_call_id: Some("call_1".into()),
                name: Some("grep".into()),
                refusal: None,
            },
        ]
    }

    #[test]
    fn appends_context_block_to_condensed_round() {
        let provider = ContextBlockProvider::new(|| "[Notepad index]\nentry-1".into());
        let mut messages = tool_round();

        condense_tool_round(&mut messages, false, Some(&provider), None);

        let text = messages
            .last()
            .unwrap()
            .content
            .as_ref()
            .unwrap()
            .as_text()
            .unwrap();
        assert!(text.contains("[Notepad index]\nentry-1"));
        assert!(text.contains("grep({\"pattern\":\"needle\"})"));
    }

    #[test]
    fn does_not_append_empty_context_block() {
        let provider = ContextBlockProvider::new(String::new);
        let mut messages = tool_round();

        condense_tool_round(&mut messages, false, Some(&provider), None);

        let text = messages
            .last()
            .unwrap()
            .content
            .as_ref()
            .unwrap()
            .as_text()
            .unwrap();
        assert!(!text.contains("Notepad index"));
    }

    #[test]
    fn message_count_alone_does_not_trigger_summarization() {
        let config = SummarizeContextConfig {
            enabled: true,
            threshold: 3,
            keep_recent: 2,
            max_chars: 10_000,
        };
        let messages = (0..20)
            .map(|index| ChatMessage::text("user", format!("small tool round {index}")))
            .collect::<Vec<_>>();

        assert!(!should_summarize(&messages, &config));
    }

    #[test]
    fn oversized_history_triggers_summarization() {
        let config = SummarizeContextConfig {
            enabled: true,
            threshold: 3,
            keep_recent: 2,
            max_chars: 100,
        };
        let messages = (0..5)
            .map(|index| {
                ChatMessage::text("user", format!("large result {index} {}", "x".repeat(50)))
            })
            .collect::<Vec<_>>();

        assert!(should_summarize(&messages, &config));
    }

    #[test]
    fn exact_tool_content_survives_long_condensed_turn_under_budget() {
        let exact_old_string = "pub fn target() {\n    return_exact_bytes();\n}";
        let config = SummarizeContextConfig {
            enabled: true,
            threshold: 3,
            keep_recent: 12,
            max_chars: 100_000,
        };
        let mut messages = (0..40)
            .map(|index| ChatMessage::text("user", format!("[Tool Results]\nround {index}")))
            .collect::<Vec<_>>();
        messages[8] = ChatMessage::text("user", format!("[Tool Results]\n{exact_old_string}"));

        assert!(!should_summarize(&messages, &config));
        assert!(
            messages
                .iter()
                .any(|message| message.content.as_ref().is_some_and(|content| {
                    content
                        .as_text()
                        .is_some_and(|text| text.contains(exact_old_string))
                }))
        );
    }
}
