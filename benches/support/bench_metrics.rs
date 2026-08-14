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
    use super::{
        ProcessEvent, ProcessEventKind, ROUTER_TOOL_NAME, StatusSubscriber, ToolCallCollector,
    };

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
}
