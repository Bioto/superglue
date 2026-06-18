//! Streaming tool-call accumulation and per-round SSE processing.

use std::collections::BTreeMap;

use crate::openai::{
    ChatCompletionChunk, FunctionCall, StreamToolCallDelta, ToolCall,
};
use crate::providers::StreamRoundOutcome;

/// Accumulates OpenAI-compat `tool_calls` deltas by stable `index`.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    by_index: BTreeMap<u32, PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_deltas(&mut self, deltas: &[StreamToolCallDelta]) {
        for delta in deltas {
            let entry = self.by_index.entry(delta.index).or_default();
            if let Some(id) = &delta.id {
                entry.id = id.clone();
            }
            if let Some(kind) = &delta.kind {
                entry.kind = kind.clone();
            }
            if let Some(func) = &delta.function {
                if let Some(name) = &func.name {
                    entry.name = name.clone();
                }
                if let Some(args) = &func.arguments {
                    entry.arguments.push_str(args);
                }
            }
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_index.is_empty()
    }

    #[must_use]
    pub fn finish(&self) -> Vec<ToolCall> {
        self.by_index
            .values()
            .map(|p| ToolCall {
                id: p.id.clone(),
                kind: if p.kind.is_empty() {
                    "function".to_string()
                } else {
                    p.kind.clone()
                },
                function: FunctionCall {
                    name: p.name.clone(),
                    arguments: p.arguments.clone(),
                },
            })
            .collect()
    }
}

/// Result of streaming one LLM round (before tool dispatch).
pub type StreamRoundResult = StreamRoundOutcome;

/// Apply one OpenAI-compat chat completion chunk to accumulators.
pub fn apply_openai_chunk(
    chunk: &ChatCompletionChunk,
    content: &mut String,
    tool_accumulator: &mut ToolCallAccumulator,
    finish_reason: &mut Option<String>,
    usage: &mut Option<crate::proto::Usage>,
) {
    if let Some(u) = &chunk.usage {
        *usage = Some(crate::proto::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
    }
    for choice in &chunk.choices {
        if let Some(fr) = &choice.finish_reason {
            *finish_reason = Some(fr.clone());
        }
        if let Some(delta) = &choice.delta.content
            && !delta.is_empty()
        {
            content.push_str(delta);
        }
        if let Some(deltas) = &choice.delta.tool_calls {
            tool_accumulator.push_deltas(deltas);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{StreamFunctionDelta, StreamToolCallDelta};

    #[test]
    fn accumulator_merges_fragments_by_index() {
        let mut acc = ToolCallAccumulator::new();
        acc.push_deltas(&[
            StreamToolCallDelta {
                index: 0,
                id: Some("call_1".into()),
                kind: Some("function".into()),
                function: Some(StreamFunctionDelta {
                    name: Some("echo".into()),
                    arguments: None,
                }),
            },
            StreamToolCallDelta {
                index: 0,
                id: None,
                kind: None,
                function: Some(StreamFunctionDelta {
                    name: None,
                    arguments: Some("{\"x\":".into()),
                }),
            },
            StreamToolCallDelta {
                index: 0,
                id: None,
                kind: None,
                function: Some(StreamFunctionDelta {
                    name: None,
                    arguments: Some("1}".into()),
                }),
            },
        ]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "echo");
        assert_eq!(calls[0].function.arguments, "{\"x\":1}");
    }
}
