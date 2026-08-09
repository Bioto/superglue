//! Anthropic Messages API SSE event accumulation.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::openai::{FunctionCall, ToolCall};
use crate::proto;

/// Maximum accumulated tool-argument JSON per stream block.
pub const PARTIAL_JSON_MAX: usize = 1024 * 1024;

#[derive(Debug, Default)]
pub struct AnthropicStreamAccumulator {
    text: String,
    thinking: String,
    thinking_signature: String,
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

/// Kind of streamed Anthropic content delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnthropicStreamDelta {
    Text(String),
    Thinking(String),
}

impl AnthropicStreamAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_sse_data(
        &mut self,
        data: &str,
    ) -> Result<Option<AnthropicStreamDelta>, serde_json::Error> {
        if data == "[DONE]" {
            return Ok(None);
        }
        let v: Value = serde_json::from_str(data)?;
        let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or_default();
        match event_type {
            "message_start" => {
                if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                    self.usage = super::anthropic::usage_from_anthropic_json(u);
                }
            }
            "content_block_start" => {
                let index = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                let block = v.get("content_block");
                let block_type = block
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or_default();
                if block_type == "tool_use" {
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
                        return Ok(Some(AnthropicStreamDelta::Text(t.to_string())));
                    }
                } else if delta_type == "thinking_delta" {
                    if let Some(t) = delta
                        .and_then(|d| d.get("thinking"))
                        .and_then(|x| x.as_str())
                    {
                        self.thinking.push_str(t);
                        return Ok(Some(AnthropicStreamDelta::Thinking(t.to_string())));
                    }
                } else if delta_type == "signature_delta" {
                    if let Some(sig) = delta
                        .and_then(|d| d.get("signature"))
                        .and_then(|x| x.as_str())
                    {
                        self.thinking_signature.push_str(sig);
                    }
                } else if delta_type == "input_json_delta" {
                    if let Some(p) = delta
                        .and_then(|d| d.get("partial_json"))
                        .and_then(|x| x.as_str())
                    {
                        let entry = self.tool_blocks.entry(index).or_default();
                        if entry.partial_json.len() + p.len() <= PARTIAL_JSON_MAX {
                            entry.partial_json.push_str(p);
                        }
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
                    self.usage = super::anthropic::usage_from_anthropic_json(u);
                }
            }
            _ => {}
        }
        Ok(None)
    }

    /// True when this SSE frame ends the message, so readers can stop instead of
    /// waiting for the provider to close the connection.
    #[must_use]
    pub fn is_terminal_sse_data(data: &str) -> bool {
        if data == "[DONE]" {
            return true;
        }
        if !data.contains("message_stop") {
            return false;
        }
        serde_json::from_str::<Value>(data)
            .ok()
            .and_then(|v| {
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|t| t == "message_stop")
            })
            .unwrap_or(false)
    }

    /// Update usage from Anthropic SSE data without accumulating content.
    pub fn apply_usage_from_sse_data(
        usage: &mut Option<proto::Usage>,
        data: &str,
    ) -> Result<(), serde_json::Error> {
        if data == "[DONE]" || !data.contains("usage") {
            return Ok(());
        }
        let v: Value = serde_json::from_str(data)?;
        let event_type = v.get("type").and_then(|t| t.as_str()).unwrap_or_default();
        match event_type {
            "message_start" => {
                if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                    *usage = super::anthropic::usage_from_anthropic_json(u);
                }
            }
            "message_delta" => {
                if let Some(u) = v.get("usage") {
                    *usage = super::anthropic::usage_from_anthropic_json(u);
                }
            }
            _ => {}
        }
        Ok(())
    }

    #[must_use]
    pub fn thinking_text(&self) -> &str {
        &self.thinking
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
        let thinking = self.thinking;
        let thinking_signature = self.thinking_signature;
        let text = self.text;
        let mut provider_blocks = Vec::new();
        if !thinking.is_empty() || !thinking_signature.is_empty() {
            let mut block = serde_json::json!({"type": "thinking", "thinking": thinking});
            if !thinking_signature.is_empty() {
                block["signature"] = serde_json::json!(thinking_signature);
            }
            provider_blocks.push(block);
        }
        if !text.is_empty() {
            provider_blocks.push(serde_json::json!({"type": "text", "text": text.clone()}));
        }
        for b in self.tool_blocks.values() {
            let input: Value =
                serde_json::from_str(&b.partial_json).unwrap_or_else(|_| serde_json::json!({}));
            provider_blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": input,
            }));
        }
        super::adapter::StreamRoundOutcome {
            content: text,
            tool_calls,
            finish_reason,
            usage: self.usage,
            provider_blocks: if provider_blocks.is_empty() {
                None
            } else {
                Some(provider_blocks)
            },
        }
    }
}
