//! Tool invocation errors (core-owned policy surface).

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

/// Errors from tool registration and invocation.
#[derive(Debug, Error, Clone)]
pub enum ToolInvokeError {
    #[error("unknown tool: {name}")]
    UnknownTool { name: String },

    #[error("duplicate tool registration: {name}")]
    DuplicateRegistration { name: String },

    #[error("tool handler failed: {message}")]
    HandlerFailed {
        message: String,
        code: Option<String>,
    },
}

impl ToolInvokeError {
    /// JSON envelope for bindings (Python/FFI); shape is stable for observability redaction tests.
    #[must_use]
    pub fn to_json(&self) -> Value {
        #[derive(Serialize)]
        struct Envelope<'a> {
            error: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            code: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool: Option<&'a str>,
        }
        match self {
            ToolInvokeError::UnknownTool { name } => serde_json::to_value(Envelope {
                error: "unknown_tool",
                code: None,
                tool: Some(name.as_str()),
            })
            .unwrap_or_else(|_| Value::String(self.to_string())),
            ToolInvokeError::DuplicateRegistration { name } => serde_json::to_value(Envelope {
                error: "duplicate_registration",
                code: None,
                tool: Some(name.as_str()),
            })
            .unwrap_or_else(|_| Value::String(self.to_string())),
            ToolInvokeError::HandlerFailed { message, code } => serde_json::to_value(Envelope {
                error: message.as_str(),
                code: code.as_deref(),
                tool: None,
            })
            .unwrap_or_else(|_| Value::String(self.to_string())),
        }
    }

    pub fn handler(message: impl Into<String>, code: Option<String>) -> Self {
        Self::HandlerFailed {
            message: message.into(),
            code,
        }
    }
}
