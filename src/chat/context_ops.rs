//! Async context optimization helpers (LLM calls live here to avoid module cycles).

use tracing::warn;

use crate::context::{
    condense_tool_round as condense_round, AaakCompressor, COMPRESS_SYSTEM, COMPRESS_USER_PREFIX,
};
use crate::context::{message_text, passthrough_aaak_context, passthrough_at_messages, transcript_from_messages};
use crate::context::SummarizeContextConfig;
use crate::http::HttpClient;
use crate::openai::{ChatMessage, MessageContent};
use crate::providers::ProviderCredentials;
use crate::tools::ToolSpec;

use super::{provider_chat_post, ChatError, ChatOptions};

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

    let name_to_spec: std::collections::HashMap<&str, &ToolSpec> = dynamic_specs
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

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

/// Compress old messages when over threshold; mutates `messages` in place.
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
    if !config.enabled || messages.len() <= config.threshold {
        return Ok(());
    }

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
                let text = m
                    .content
                    .as_ref()
                    .and_then(|c| c.as_text())
                    .unwrap_or("");
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

        let provider = crate::providers::resolve_provider(
            &crate::providers::parse_model_ref(summarize_model),
        );
        let normalized = provider.parse_chat_response(&val).map_err(|e| {
            ChatError::Http(crate::http::Error::InvalidJson(e.to_string()))
        })?;
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
    let normalized = provider.parse_chat_response(&val).map_err(|e| {
        ChatError::Http(crate::http::Error::InvalidJson(e.to_string()))
    })?;

    Ok(normalized.content.unwrap_or_default().trim().to_string())
}

pub fn condense_tool_round(messages: &mut Vec<ChatMessage>, aaak_tool_condensing: bool) {
    condense_round(messages, aaak_tool_condensing);
}
