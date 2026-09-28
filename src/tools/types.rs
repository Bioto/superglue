//! Tool specification and invocation envelopes (JSON).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::proto;

/// Whether a tool's output may be replaced by a notepad offload stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolContextPolicy {
    /// Output is the state the model asked for — never offload.
    Inline,
    /// Bulk payload; offload when over the configured size threshold. (default)
    #[default]
    OffloadWhenLarge,
    /// Always offload when extractable content exists.
    AlwaysOffload,
}

impl ToolContextPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::OffloadWhenLarge => "offload_when_large",
            Self::AlwaysOffload => "always_offload",
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "inline" => Some(Self::Inline),
            "offload_when_large" => Some(Self::OffloadWhenLarge),
            "always_offload" => Some(Self::AlwaysOffload),
            _ => None,
        }
    }
}

/// Registered tool metadata: name plus opaque JSON Schema for parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    /// JSON Schema describing parameters (opaque to the core).
    pub parameters_schema: Value,
    /// Optional human-readable description shown to the model.
    pub description: Option<String>,
    /// When true, always included in dynamic tool mode (GlueLLM `@static_tool` parity).
    #[serde(default)]
    pub static_tool: bool,
}

impl ToolSpec {
    #[must_use]
    pub fn new(name: impl Into<String>, parameters_schema: Value) -> Self {
        Self {
            name: name.into(),
            parameters_schema,
            description: None,
            static_tool: false,
        }
    }

    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    #[must_use]
    pub fn with_static_tool(mut self, static_tool: bool) -> Self {
        self.static_tool = static_tool;
        self
    }
}

/// Single tool call emitted by an assistant/model (simulated in the harness).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolInvocation {
    pub tool_name: String,
    pub arguments: Value,
}

impl From<proto::ToolSpec> for ToolSpec {
    fn from(p: proto::ToolSpec) -> Self {
        let parameters_schema: Value = serde_json::from_str(&p.parameters_schema_json)
            .unwrap_or(Value::Object(Default::default()));
        ToolSpec {
            name: p.name,
            parameters_schema,
            description: p.description,
            static_tool: p.static_tool,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tool error policy
// ---------------------------------------------------------------------------

/// What the chat loop should do when a tool's handler returns an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnToolError {
    /// Propagate the error immediately, aborting the run. (default)
    FailFast,
    /// Append the error as the tool result and let the model continue.
    Skip,
    /// Retry the tool call up to `max` times with exponential backoff before
    /// falling back to [`FailFast`](OnToolError::FailFast).
    Retry {
        /// Maximum number of retries (does not count the original attempt).
        max: u32,
        /// Base delay in milliseconds for the first retry. Doubles each attempt.
        initial_delay_ms: u64,
    },
}

impl Default for OnToolError {
    fn default() -> Self {
        OnToolError::FailFast
    }
}

/// Per-tool error policy attached to a registered tool.
///
/// Obtain the default (fail-fast) policy with [`ToolRetryPolicy::default()`].
#[derive(Debug, Clone, Default)]
pub struct ToolRetryPolicy {
    /// How the chat loop handles [`ToolInvokeError::HandlerFailed`] for this tool.
    pub on_error: OnToolError,
}
