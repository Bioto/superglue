//! MCP client bridge — connect MCP servers and register tools into [`ToolRegistry`].

pub mod config;
pub mod runtime;

pub use config::{
    default_mcp_servers, enabled_servers, ensure_servers_file, format_config_status,
    load_servers_file, load_servers_file_or_default, load_status_file, record_server_status,
    resolve_env_value, resolve_headers, save_servers_file, save_status_file, McpAuthKind,
    McpServerEntry, McpServerStatus, McpServersFile, McpServersStatusFile, McpToolPolicy,
    DEFAULT_CONTEXT7_MCP_URL, DEFAULT_GMAIL_MCP_URL,
};
pub use runtime::{
    connect_enabled_servers, McpConnectOptions, McpConnectSummary, McpConnections,
    McpOAuthProvider,
};

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, JsonObject};
use rmcp::service::{RoleClient, RunningService, ServiceError};
use rmcp::transport::{
    streamable_http_client::StreamableHttpClientTransportConfig, ConfigureCommandExt,
    StreamableHttpClientTransport, TokioChildProcess,
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
#[derive(Debug, Clone, Default)]
pub struct McpHttpConfig {
    pub url: String,
    pub label: Option<String>,
    /// Bearer token sent as `Authorization` on MCP HTTP requests.
    pub auth_header: Option<String>,
    /// Additional HTTP headers (e.g. `CONTEXT7_API_KEY`).
    pub custom_headers: HashMap<String, String>,
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
        let mut transport_config = StreamableHttpClientTransportConfig::with_uri(config.url.clone());
        if let Some(token) = config.auth_header {
            transport_config = transport_config.auth_header(token);
        }
        if !config.custom_headers.is_empty() {
            use http::{HeaderName, HeaderValue};
            let mut headers = HashMap::new();
            for (name, value) in config.custom_headers {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|e| McpError::Connect(format!("invalid header name {name}: {e}")))?;
                let value = HeaderValue::from_str(&value)
                    .map_err(|e| McpError::Connect(format!("invalid header value for {name}: {e}")))?;
                headers.insert(name, value);
            }
            transport_config = transport_config.custom_headers(headers);
        }
        let transport = StreamableHttpClientTransport::from_config(transport_config);
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
        self.register_tools_if(registry, prefix, |_| true).await
    }

    /// Register MCP tools whose server-side name passes `include`.
    pub async fn register_tools_if(
        self: &Arc<Self>,
        registry: &ToolRegistry,
        prefix: Option<&str>,
        include: impl Fn(&str) -> bool,
    ) -> Result<Vec<String>, McpError> {
        let tools = {
            let guard = self.inner.lock().await;
            guard.list_all_tools().await?
        };
        let prefix = prefix.unwrap_or(&self.label);
        let mut registered = Vec::new();
        for tool in tools {
            let mcp_name = tool.name.as_ref();
            if !include(mcp_name) {
                continue;
            }
            let reg_name = prefixed_tool_name(prefix, mcp_name);
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

    /// Invoke a tool on this MCP session (for lazy proxy wrappers).
    pub async fn invoke_tool(
        &self,
        mcp_name: &str,
        arguments: Value,
    ) -> Result<Value, crate::tools::ToolInvokeError> {
        let result = self
            .call_mcp_tool(mcp_name, arguments)
            .await
            .map_err(|e| crate::tools::ToolInvokeError::handler(e.to_string(), None))?;
        call_tool_result_to_value(result)
            .map_err(|e| crate::tools::ToolInvokeError::handler(e.to_string(), None))
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
        let params = match args_obj {
            Some(args) => CallToolRequestParams::new(mcp_name.to_string()).with_arguments(args),
            None => CallToolRequestParams::new(mcp_name.to_string()),
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
        static_tool: false,
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

fn content_to_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
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
    fn mcp_http_config_carries_auth_header() {
        let config = McpHttpConfig {
            url: "https://example.com/mcp".into(),
            label: Some("gmail".into()),
            auth_header: Some("access-token".into()),
            custom_headers: HashMap::new(),
        };
        assert_eq!(config.auth_header.as_deref(), Some("access-token"));
        let transport_config =
            StreamableHttpClientTransportConfig::with_uri(config.url.clone())
                .auth_header(config.auth_header.unwrap());
        assert_eq!(
            transport_config.auth_header.as_deref(),
            Some("access-token")
        );
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
