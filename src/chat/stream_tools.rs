//! Streaming tool-call accumulation and per-round SSE processing.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::openai::{ChatCompletionChunk, FunctionCall, StreamToolCallDelta, ToolCall};

/// Accumulates OpenAI-compat `tool_calls` deltas by stable `index`.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    by_index: std::collections::BTreeMap<u32, PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

impl PartialToolCall {
    fn to_tool_call(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            kind: if self.kind.is_empty() {
                "function".to_string()
            } else {
                self.kind.clone()
            },
            function: FunctionCall {
                name: self.name.clone(),
                arguments: self.arguments.clone(),
            },
        }
    }
}

fn partial_is_ready(partial: &PartialToolCall, boundary: bool) -> bool {
    if partial.name.is_empty() {
        return false;
    }
    let raw = partial.arguments.trim();
    if raw.is_empty() {
        return boundary;
    }
    serde_json::from_str::<Value>(raw).is_ok() || boundary
}

impl ToolCallAccumulator {
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn max_index(&self) -> Option<u32> {
        self.by_index.keys().next_back().copied()
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
    pub fn indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.by_index.keys().copied()
    }

    #[must_use]
    pub fn finish(&self) -> Vec<ToolCall> {
        self.by_index
            .values()
            .map(PartialToolCall::to_tool_call)
            .collect()
    }

    fn partial_at(&self, index: u32) -> Option<&PartialToolCall> {
        self.by_index.get(&index)
    }
}

/// Tracks streaming tool-call fragments and yields calls ready for early dispatch.
#[derive(Debug, Default)]
pub struct StreamingToolDispatch {
    accumulator: ToolCallAccumulator,
    dispatched: BTreeSet<u32>,
}

impl StreamingToolDispatch {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply SSE deltas and return tool calls that became ready during this chunk.
    pub fn push_deltas(&mut self, deltas: &[StreamToolCallDelta]) -> Vec<ToolCall> {
        let prior_max = self.accumulator.max_index();
        self.accumulator.push_deltas(deltas);
        let new_max = self.accumulator.max_index();
        let mut ready = Vec::new();

        if let (Some(prior), Some(current)) = (prior_max, new_max)
            && current > prior
        {
            for index in 0..current {
                if let Some(tc) = self.take_if_ready(index, true) {
                    ready.push(tc);
                }
            }
        }

        if let Some(max_index) = new_max {
            for index in 0..=max_index {
                if let Some(tc) = self.take_if_ready(index, false) {
                    ready.push(tc);
                }
            }
        }

        ready
    }

    /// Return any remaining tool calls at the end of an LLM stream round.
    pub fn drain_at_round_end(&mut self) -> Vec<ToolCall> {
        let indices: Vec<u32> = self.accumulator.indices().collect();
        let mut ready = Vec::new();
        for index in indices {
            if let Some(tc) = self.take_if_ready(index, true) {
                ready.push(tc);
            }
        }
        ready
    }

    #[must_use]
    pub fn finish_remaining(&self) -> Vec<ToolCall> {
        self.accumulator.finish()
    }

    fn take_if_ready(&mut self, index: u32, boundary: bool) -> Option<ToolCall> {
        if self.dispatched.contains(&index) {
            return None;
        }
        let partial = self.accumulator.partial_at(index)?;
        if !partial_is_ready(partial, boundary) {
            return None;
        }
        self.dispatched.insert(index);
        Some(partial.to_tool_call())
    }
}

/// Apply one OpenAI-compat chat completion chunk to accumulators.
pub fn apply_openai_chunk(
    chunk: &ChatCompletionChunk,
    content: &mut String,
    tool_dispatch: &mut StreamingToolDispatch,
    finish_reason: &mut Option<String>,
    usage: &mut Option<crate::proto::Usage>,
) -> Vec<ToolCall> {
    if let Some(u) = &chunk.usage {
        *usage = Some(crate::proto::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });
    }
    let mut ready = Vec::new();
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
            ready.extend(tool_dispatch.push_deltas(deltas));
        }
    }
    ready
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

    #[test]
    fn streaming_dispatch_emits_complete_json_before_round_end() {
        let mut dispatch = StreamingToolDispatch::new();
        let ready = dispatch.push_deltas(&[StreamToolCallDelta {
            index: 0,
            id: Some("call_1".into()),
            kind: Some("function".into()),
            function: Some(StreamFunctionDelta {
                name: Some("echo".into()),
                arguments: Some("{\"x\":1}".into()),
            }),
        }]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].function.name, "echo");
        assert!(dispatch.drain_at_round_end().is_empty());
    }

    #[test]
    fn streaming_dispatch_boundary_completes_prior_index() {
        let mut dispatch = StreamingToolDispatch::new();
        dispatch.push_deltas(&[StreamToolCallDelta {
            index: 0,
            id: Some("call_1".into()),
            kind: Some("function".into()),
            function: Some(StreamFunctionDelta {
                name: Some("a".into()),
                arguments: Some("{\"x\":".into()),
            }),
        }]);
        let ready = dispatch.push_deltas(&[StreamToolCallDelta {
            index: 1,
            id: Some("call_2".into()),
            kind: Some("function".into()),
            function: Some(StreamFunctionDelta {
                name: Some("b".into()),
                arguments: None,
            }),
        }]);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].function.name, "a");
    }
}
