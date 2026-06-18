//! MCP client bridge — connect MCP servers and register tools into [`ToolRegistry`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rmcp::model::{CallToolRequestParams, CallToolResult, Content, JsonObject};
use rmcp::service::{RoleClient, RunningService, ServiceError};
use rmcp::transport::{
    ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess,
};
use rmcp::{ServiceExt, model::Tool as McpToolDef};
use serde_json::{json, Map, Value};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

/// Errors from MCP connection and tool invocation.
#[derive(Debug, Error)]
pub enum McpError {
    #[error("MCP connect failed: {0}")]
    Connect(String),
    #[error(transparent)]
    Service(#[from] ServiceError),
    #[error(transparent)]
    Tool(#[from] crate::tools::ToolInvokeError),
    #[error("MCP tool error: {0}")]
    ToolFailed(String),
}

/// Stdio MCP server launch configuration.
#[derive(Debug, Clone)]
pub struct McpStdioConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: Option<HashMap<String, String>>,
    pub label: Option<String>,
}

/// HTTP (streamable) MCP server URL.
#[derive(Debug, Clone)]
pub struct McpHttpConfig {
    pub url: String,
    pub label: Option<String>,
}

/// Active MCP session; disconnect via [`McpSession::close`] or on drop.
pub struct McpSession {
    inner: Mutex<RunningService<RoleClient, ()>>,
    label: String,
}

impl McpSession {
    /// Connect to an MCP server over stdio (child process).
    pub async fn connect_stdio(config: McpStdioConfig) -> Result<Arc<Self>, McpError> {
        let label = config
            .label
            .clone()
            .unwrap_or_else(|| default_label(&config.command));
        let cmd = config.command.clone();
        let args = config.args.clone();
        let env = config.env.clone();
        let transport = TokioChildProcess::new(tokio::process::Command::new(&config.command).configure(
            |c| {
                for a in &args {
                    c.arg(a);
                }
                if let Some(vars) = &env {
                    for (k, v) in vars {
                        c.env(k, v);
                    }
                }
            },
        ))
        .map_err(|e| McpError::Connect(format!("spawn {cmd}: {e}")))?;

        let client = ()
            .serve(transport)
            .await
            .map_err(|e| McpError::Connect(e.to_string()))?;

        Ok(Arc::new(Self {
            inner: Mutex::new(client),
            label,
        }))
    }

    /// Connect to an MCP server over streamable HTTP.
    pub async fn connect_http(config: McpHttpConfig) -> Result<Arc<Self>, McpError> {
        let label = config
            .label
            .clone()
            .unwrap_or_else(|| "mcp_http".to_string());
        let transport = StreamableHttpClientTransport::from_uri(config.url.clone());
        let client = ()
            .serve(transport)
            .await
            .map_err(|e| McpError::Connect(e.to_string()))?;

        Ok(Arc::new(Self {
            inner: Mutex::new(client),
            label,
        }))
    }

    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// List tools from the server and register each as [`McpTool`] in `registry`.
    pub async fn register_tools(
        self: &Arc<Self>,
        registry: &ToolRegistry,
        prefix: Option<&str>,
    ) -> Result<Vec<String>, McpError> {
        let tools = {
            let guard = self.inner.lock().await;
            guard.list_all_tools().await?
        };
        let prefix = prefix.unwrap_or(&self.label);
        let mut registered = Vec::new();
        for tool in tools {
            let reg_name = prefixed_tool_name(prefix, tool.name.as_ref());
            let mcp_tool = McpTool {
                session: Arc::clone(self),
                registered_name: reg_name.clone(),
                mcp_name: tool.name.to_string(),
                spec: mcp_tool_to_spec(&tool, &reg_name),
            };
            registry.register(Arc::new(mcp_tool)).await?;
            registered.push(reg_name);
        }
        Ok(registered)
    }

    /// Gracefully close the MCP connection.
    pub async fn close(&self) -> Result<(), McpError> {
        let mut guard = self.inner.lock().await;
        guard
            .close()
            .await
            .map_err(|e| McpError::Connect(format!("close: {e}")))?;
        Ok(())
    }

    async fn call_mcp_tool(
        &self,
        mcp_name: &str,
        arguments: Value,
    ) -> Result<CallToolResult, McpError> {
        let args_obj: Option<JsonObject> = match arguments {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => Some(Map::from_iter([(
                "value".to_string(),
                other,
            )])),
        };
        let params = CallToolRequestParams {
            meta: None,
            name: mcp_name.to_string().into(),
            arguments: args_obj,
            task: None,
        };
        let guard = self.inner.lock().await;
        guard.peer().call_tool(params).await.map_err(McpError::from)
    }
}

#[must_use]
pub fn prefixed_tool_name(prefix: &str, tool_name: &str) -> String {
    format!("{}__{}", prefix, tool_name)
}

fn default_label(command: &str) -> String {
    command
        .rsplit('/')
        .next()
        .unwrap_or(command)
        .to_string()
}

fn mcp_tool_to_spec(tool: &McpToolDef, registered_name: &str) -> ToolSpec {
    ToolSpec {
        name: registered_name.to_string(),
        description: tool.description.as_ref().map(|d| d.to_string()),
        parameters_schema: Value::Object(tool.input_schema.as_ref().clone()),
    }
}

fn call_tool_result_to_value(result: CallToolResult) -> Result<Value, McpError> {
    if result.is_error == Some(true) {
        let msg = result
            .structured_content
            .map(|v| v.to_string())
            .unwrap_or_else(|| content_to_text(&result.content));
        return Err(McpError::ToolFailed(msg));
    }
    if let Some(structured) = result.structured_content {
        return Ok(structured);
    }
    let text = content_to_text(&result.content);
    if text.is_empty() {
        Ok(json!({}))
    } else {
        Ok(json!({ "content": text }))
    }
}

fn content_to_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_ref()))
        .collect::<Vec<_>>()
        .join("")
}

struct McpTool {
    session: Arc<McpSession>,
    registered_name: String,
    mcp_name: String,
    spec: ToolSpec,
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        let result = self
            .session
            .call_mcp_tool(&self.mcp_name, arguments)
            .await
            .map_err(|e| ToolInvokeError::handler(e.to_string(), Some(self.registered_name.clone())))?;
        call_tool_result_to_value(result).map_err(|e| {
            ToolInvokeError::handler(e.to_string(), Some(self.registered_name.clone()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_tool_name_avoids_collisions() {
        assert_eq!(prefixed_tool_name("srv", "echo"), "srv__echo");
        assert_eq!(prefixed_tool_name("a", "b"), "a__b");
    }

    #[test]
    fn mcp_tool_to_spec_uses_registered_name() {
        let schema: rmcp::model::JsonObject =
            serde_json::from_value(json!({"type": "object"})).unwrap();
        let tool = McpToolDef::new("echo", "echo args", schema);
        let spec = mcp_tool_to_spec(&tool, "srv__echo");
        assert_eq!(spec.name, "srv__echo");
        assert_eq!(spec.description.as_deref(), Some("echo args"));
    }
}
