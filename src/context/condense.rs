//! Tool-round message condensing (GlueLLM `condense_tool_messages` parity).

use crate::openai::{ChatMessage, MessageContent, ToolCall};

use super::aaak::{AaakCompressor, CONDENSE_ANTI_LOOP_SUFFIX};

/// Cap argument previews in condensed summaries so large patch bodies cannot bloat context.
const CONDENSE_ARGS_MAX_CHARS: usize = 240;

fn truncate_args_preview(args: &str) -> String {
    let trimmed = args.trim();
    if trimmed.chars().count() <= CONDENSE_ARGS_MAX_CHARS {
        return trimmed.to_string();
    }
    let preview: String = trimmed.chars().take(CONDENSE_ARGS_MAX_CHARS).collect();
    format!("{preview}…")
}

/// Replace the last pure tool-call round with a single condensed user message.
///
/// Only condenses when the assistant message has tool calls and no visible text content.
pub fn condense_tool_round(messages: &mut Vec<ChatMessage>, aaak_tool_condensing: bool) {
    let mut tool_response_indices: Vec<usize> = Vec::new();
    let mut idx = messages.len().saturating_sub(1);
    while idx < messages.len() && messages[idx].role == "tool" {
        tool_response_indices.push(idx);
        if idx == 0 {
            break;
        }
        idx -= 1;
    }
    tool_response_indices.reverse();

    if tool_response_indices.is_empty() {
        return;
    }

    let assistant_idx = if tool_response_indices[0] == 0 {
        return;
    } else {
        tool_response_indices[0] - 1
    };

    let assistant_msg = &messages[assistant_idx];
    if assistant_msg.role != "assistant" {
        return;
    }

    let tool_calls: Vec<ToolCall> = assistant_msg
        .tool_calls
        .as_ref()
        .map(|t| t.to_vec())
        .unwrap_or_default();
    if tool_calls.is_empty() {
        return;
    }

    if assistant_msg
        .content
        .as_ref()
        .and_then(|c| c.as_text())
        .is_some_and(|t| !t.is_empty())
    {
        return;
    }

    // Keep name + arguments so the model still knows *what* it already called
    // after we drain the assistant message (which owns `tool_calls`).
    let mut id_to_call: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    for tc in &tool_calls {
        id_to_call.insert(
            tc.id.clone(),
            (tc.function.name.clone(), tc.function.arguments.clone()),
        );
    }

    let condensed_content = if aaak_tool_condensing {
        if messages.first().is_some_and(|m| m.role == "system") {
            if let Some(sys) = messages.first_mut() {
                AaakCompressor::ensure_preamble_in_system(sys);
            }
        }
        let tool_messages: Vec<ChatMessage> = tool_response_indices
            .iter()
            .map(|&i| messages[i].clone())
            .collect();
        let id_to_name: std::collections::HashMap<String, String> = id_to_call
            .iter()
            .map(|(id, (name, _))| (id.clone(), name.clone()))
            .collect();
        AaakCompressor::encode_tool_round(&tool_calls, &tool_messages, &id_to_name)
    } else {
        let mut lines = vec!["[Tool Results]".to_string()];
        for &ti in &tool_response_indices {
            let tool_msg = &messages[ti];
            let tc_id = tool_msg.tool_call_id.as_deref().unwrap_or("");
            let (name, args) = id_to_call
                .get(tc_id)
                .map(|(n, a)| (n.as_str(), a.as_str()))
                .unwrap_or((tc_id, ""));
            let args_preview = truncate_args_preview(args);
            let result = tool_msg
                .content
                .as_ref()
                .and_then(|c| c.as_text())
                .unwrap_or("");
            lines.push(format!("- {name}({args_preview}) -> {result}"));
        }
        lines.join("\n")
    };

    let condensed_content = format!("{condensed_content}{CONDENSE_ANTI_LOOP_SUFFIX}");

    messages.drain(assistant_idx..);
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Text(condensed_content)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
        provider_blocks: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::FunctionCall;

    fn assistant_with_tools() -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "echo".into(),
                    arguments: r#"{"x":1}"#.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
            refusal: None,
            provider_blocks: None,
        }
    }

    #[test]
    fn condense_replaces_tool_round_with_user_summary() {
        let mut messages = vec![
            assistant_with_tools(),
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(r#"{"ok":true}"#.into())),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
                name: Some("echo".into()),
                refusal: None,
                provider_blocks: None,
            },
        ];
        condense_tool_round(&mut messages, false);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert!(
            messages[0]
                .content
                .as_ref()
                .and_then(|c| c.as_text())
                .unwrap()
                .contains("[Tool Results]")
        );
        let text = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .unwrap();
        assert!(text.contains("echo({\"x\":1})"));
        assert!(text.contains("do not re-invoke these same tool calls"));
        assert!(text.contains("you may call new tools"));
        assert!(!text.contains("do not call tools"));
    }

    #[test]
    fn condense_preserves_tool_call_arguments() {
        let mut messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "g1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "grep".into(),
                        arguments: r#"{"pattern":"condense_tool_round","path":"src/context"}"#
                            .into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
                refusal: None,
                provider_blocks: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(r#"{"match_count":0}"#.into())),
                tool_calls: None,
                tool_call_id: Some("g1".into()),
                name: Some("grep".into()),
                refusal: None,
                provider_blocks: None,
            },
        ];
        condense_tool_round(&mut messages, false);
        let text = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .unwrap();
        assert!(text.contains("grep("));
        assert!(text.contains("condense_tool_round"));
        assert!(text.contains("src/context"));
        assert!(text.contains(r#"{"match_count":0}"#));
    }

    #[test]
    fn condense_truncates_long_arguments() {
        let long_args = format!(
            r#"{{"patch":"{}"}}"#,
            "x".repeat(CONDENSE_ARGS_MAX_CHARS + 80)
        );
        let mut messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "p1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "patch".into(),
                        arguments: long_args,
                    },
                }]),
                tool_call_id: None,
                name: None,
                refusal: None,
                provider_blocks: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(r#"{"ok":true}"#.into())),
                tool_calls: None,
                tool_call_id: Some("p1".into()),
                name: Some("patch".into()),
                refusal: None,
                provider_blocks: None,
            },
        ];
        condense_tool_round(&mut messages, false);
        let text = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .unwrap();
        assert!(text.contains("patch("));
        assert!(text.contains('…'));
        let args_span = text
            .split("patch(")
            .nth(1)
            .and_then(|rest| rest.split(") ->").next())
            .unwrap_or("");
        assert!(args_span.chars().count() <= CONDENSE_ARGS_MAX_CHARS + 1);
    }

    #[test]
    fn condense_after_ask_user_allows_follow_up_tools() {
        let mut messages = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "ask_user".into(),
                        arguments: r#"{"title":"Commit?"}"#.into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
                refusal: None,
                provider_blocks: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(
                    r#"{"answers":{"commit":"Use this message and commit"}}"#.into(),
                )),
                tool_calls: None,
                tool_call_id: Some("c1".into()),
                name: Some("ask_user".into()),
                refusal: None,
                provider_blocks: None,
            },
        ];
        condense_tool_round(&mut messages, false);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        let text = messages[0]
            .content
            .as_ref()
            .and_then(|c| c.as_text())
            .unwrap();
        assert!(text.contains("ask_user"));
        assert!(text.contains("Use this message and commit"));
        assert!(text.contains("you may call new tools"));
        assert!(!text.contains("do not call tools"));
    }
}
