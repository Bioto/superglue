//! Anthropic Messages API SSE event accumulation.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::openai::{FunctionCall, ToolCall};
use crate::proto;

#[derive(Debug, Default)]
pub struct AnthropicStreamAccumulator {
    text: String,
    tool_blocks: BTreeMap<u32, PartialToolBlock>,
    finish_reason: Option<String>,
    usage: Option<proto::Usage>,
}

#[derive(Debug, Default)]
struct PartialToolBlock {
    id: String,
    name: String,
    partial_json: String,
}

impl AnthropicStreamAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_sse_data(&mut self, data: &str) -> Result<Option<String>, serde_json::Error> {
        if data == "[DONE]" {
            return Ok(None);
        }
        let v: Value = serde_json::from_str(data)?;
        let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or_default();
        match event_type {
            "message_start" => {
                if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                    let input = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                    let output =
                        u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                    self.usage = Some(crate::usage::usage_from_breakdown(
                        crate::usage::UsageBreakdown::from_counts(input, output),
                    ));
                }
            }
            "content_block_start" => {
                let index = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let block = v.get("content_block");
                if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
                    let entry = self.tool_blocks.entry(index).or_default();
                    entry.id = block
                        .and_then(|b| b.get("id"))
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string();
                    entry.name = block
                        .and_then(|b| b.get("name"))
                        .and_then(|x| x.as_str())
                        .unwrap_or_default()
                        .to_string();
                }
            }
            "content_block_delta" => {
                let index = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let delta = v.get("delta");
                let delta_type = delta
                    .and_then(|d| d.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if delta_type == "text_delta" {
                    if let Some(t) = delta.and_then(|d| d.get("text")).and_then(|x| x.as_str()) {
                        self.text.push_str(t);
                        return Ok(Some(t.to_string()));
                    }
                } else if delta_type == "input_json_delta" {
                    if let Some(p) = delta
                        .and_then(|d| d.get("partial_json"))
                        .and_then(|x| x.as_str())
                    {
                        let entry = self.tool_blocks.entry(index).or_default();
                        entry.partial_json.push_str(p);
                    }
                }
            }
            "message_delta" => {
                if let Some(sr) = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|x| x.as_str())
                {
                    self.finish_reason = Some(sr.to_string());
                }
                if let Some(u) = v.get("usage") {
                    let input = u.get("input_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                    let output =
                        u.get("output_tokens").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                    self.usage = Some(crate::usage::usage_from_breakdown(
                        crate::usage::UsageBreakdown::from_counts(input, output),
                    ));
                }
            }
            _ => {}
        }
        Ok(None)
    }

    #[must_use]
    pub fn into_round_outcome(self) -> super::adapter::StreamRoundOutcome {
        let tool_calls: Vec<ToolCall> = self
            .tool_blocks
            .values()
            .map(|b| ToolCall {
                id: b.id.clone(),
                kind: "function".to_string(),
                function: FunctionCall {
                    name: b.name.clone(),
                    arguments: b.partial_json.clone(),
                },
            })
            .collect();
        let finish_reason = if !tool_calls.is_empty() {
            Some("tool_calls".to_string())
        } else {
            self.finish_reason.map(|sr| {
                if sr == "tool_use" {
                    "tool_calls".to_string()
                } else {
                    "stop".to_string()
                }
            })
        };
        super::adapter::StreamRoundOutcome {
            content: self.text,
            tool_calls,
            finish_reason,
            usage: self.usage,
        }
    }
}
