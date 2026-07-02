//! Outcome metrics shared by wiremock and live benchmarks.

use std::collections::HashSet;

use superglue::openai::ChatMessage;
use superglue::tools::ROUTER_TOOL_NAME;

/// Count non-router tool calls in caller-visible messages.
#[must_use]
pub fn count_tool_calls(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .flat_map(|m| m.tool_calls.iter().flatten())
        .filter(|tc| tc.function.name != ROUTER_TOOL_NAME)
        .count() as u32
}

/// Whether every expected tool name appears in tool calls or condensed message text.
#[must_use]
pub fn completed_expected(messages: &[ChatMessage], expected: &[&str]) -> bool {
    let observed = tools_observed_in_messages(messages);
    expected.iter().all(|name| observed.contains(*name))
}

fn tools_observed_in_messages(messages: &[ChatMessage]) -> HashSet<String> {
    let mut names = HashSet::new();
    for msg in messages {
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                if tc.function.name != ROUTER_TOOL_NAME {
                    names.insert(tc.function.name.clone());
                }
            }
        }
        let text = message_text(msg);
        for token in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if !token.is_empty() {
                names.insert(token.to_string());
            }
        }
    }
    names
}

fn message_text(msg: &ChatMessage) -> String {
    match &msg.content {
        Some(superglue::openai::MessageContent::Text(t)) => t.clone(),
        Some(superglue::openai::MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                superglue::openai::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use superglue::openai::{FunctionCall, ToolCall};

    #[test]
    fn counts_non_router_tool_calls() {
        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: "{}".into(),
                },
            }]),
            tool_call_id: None,
            name: None,
            refusal: None,
        }];
        assert_eq!(count_tool_calls(&messages), 1);
    }
}
