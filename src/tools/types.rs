//! Tool specification and invocation envelopes (JSON).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::proto;

/// Registered tool metadata: name plus opaque JSON Schema for parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    /// JSON Schema describing parameters (opaque to the core).
    pub parameters_schema: Value,
    /// Optional human-readable description shown to the model.
    pub description: Option<String>,
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
        }
    }
}
