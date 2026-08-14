//! Outcome metrics shared by wiremock and live benchmarks.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use superglue::events::{ProcessEvent, ProcessEventKind, StatusSubscriber};
use superglue::openai::ChatMessage;
use superglue::tools::ROUTER_TOOL_NAME;

/// Counts caller-visible tool calls from lifecycle events, including calls
/// whose assistant messages are removed by tool-result condensing.
pub struct ToolCallCollector {
    count: AtomicU32,
}

impl ToolCallCollector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: AtomicU32::new(0),
        }
    }

    #[must_use]
    pub fn count(&self) -> u32 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for ToolCallCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StatusSubscriber for ToolCallCollector {
    async fn on_event(&self, event: ProcessEvent) {
        if event.kind == ProcessEventKind::ToolCallStart
            && event
                .metadata
                .get("tool_name")
                .is_none_or(|name| name != ROUTER_TOOL_NAME)
        {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether every expected tool was actually called or appears in a runtime
/// condensed representation.
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
                if tc.function.name == ROUTER_TOOL_NAME {
                    continue;
                }
                if tc.function.name != "code" {
                    names.insert(tc.function.name.clone());
                }
            }
        }

        if msg.role == "tool" {
            if let Some(name) = &msg.name {
                if name != ROUTER_TOOL_NAME && name != "code" {
                    names.insert(name.clone());
                }
            }
            observe_code_result(&message_text(msg), &mut names);
        } else {
            observe_condensed_text(&message_text(msg), &mut names);
        }
    }
    names
}

fn observe_code_result(text: &str, names: &mut HashSet<String>) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let Some(calls) = value.get("calls").and_then(|calls| calls.as_array()) else {
        return;
    };
    for call in calls {
        if let Some(name) = call.get("tool").and_then(|tool| tool.as_str())
            && name != ROUTER_TOOL_NAME
            && name != "code"
        {
            names.insert(name.to_string());
        }
    }
}

fn observe_condensed_text(text: &str, names: &mut HashSet<String>) {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("- ") {
            observe_name_before_call(rest, names);
            if let Some(result) = rest.split_once(" -> ").map(|(_, result)| result) {
                observe_code_result(result, names);
            }
        }
        let mut rest = line;
        while let Some(offset) = rest.find("T:") {
            rest = &rest[offset + 2..];
            observe_name_before_call(rest, names);
            let name_len = rest
                .find(|c: char| c == '(' || c.is_whitespace() || c == '|')
                .unwrap_or(rest.len());
            if name_len == 0 {
                break;
            }
            rest = &rest[name_len..];
        }
    }
}

fn observe_name_before_call(text: &str, names: &mut HashSet<String>) {
    let name_len = text.find('(').unwrap_or(0);
    if name_len > 0 {
        let name = text[..name_len].trim();
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            names.insert(name.to_string());
        }
    }
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
    use super::{
        ProcessEvent, ProcessEventKind, ROUTER_TOOL_NAME, StatusSubscriber, ToolCallCollector,
        completed_expected,
    };
    use superglue::openai::ChatMessage;

    #[tokio::test]
    async fn event_collector_counts_condensed_tool_calls() {
        let collector = ToolCallCollector::new();
        let mut weather = ProcessEvent::new(ProcessEventKind::ToolCallStart, "req", "mock");
        weather
            .metadata
            .insert("tool_name".into(), "get_weather".into());
        collector.on_event(weather).await;

        let mut router = ProcessEvent::new(ProcessEventKind::ToolCallStart, "req", "mock");
        router
            .metadata
            .insert("tool_name".into(), ROUTER_TOOL_NAME.into());
        collector.on_event(router).await;

        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn completion_does_not_count_tool_names_in_final_prose() {
        let messages = vec![ChatMessage::text(
            "assistant",
            "I could use get_weather, but this answer is from memory.",
        )];

        assert!(!completed_expected(&messages, &["get_weather"]));
    }

    #[test]
    fn completion_counts_condensed_tool_results() {
        let messages = vec![ChatMessage::text(
            "user",
            "[AT]\nT:get_weather()→city=Paris | T:get_forecast()→days=5",
        )];

        assert!(completed_expected(
            &messages,
            &["get_weather", "get_forecast"]
        ));
    }

    #[test]
    fn completion_counts_nested_code_calls() {
        let messages = vec![ChatMessage::text(
            "tool",
            r#"{"ok":true,"result":{"temp":72},"calls":[{"tool":"get_weather","ok":true}]}"#,
        )];

        assert!(completed_expected(&messages, &["get_weather"]));
    }
}
