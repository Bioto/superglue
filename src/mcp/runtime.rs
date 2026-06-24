//! Connect MCP servers from [`McpServersFile`] configuration.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;

use crate::tools::ToolRegistry;

use super::config::{
    enabled_servers, record_server_status, resolve_headers, McpAuthKind, McpServerEntry,
    McpServerStatus, McpServersFile, McpToolPolicy,
};
use super::{McpError, McpHttpConfig, McpSession};

pub struct McpConnections {
    pub sessions: Vec<Arc<McpSession>>,
    pub tool_count: usize,
}

pub struct McpConnectSummary {
    pub sessions: Vec<Arc<McpSession>>,
    pub connected_ids: Vec<String>,
}

pub struct McpConnectOptions<'a> {
    pub config: &'a McpServersFile,
    pub full_registry: &'a ToolRegistry,
    pub readonly_registry: &'a ToolRegistry,
    pub oauth: Option<&'a dyn McpOAuthProvider>,
    pub status_path: Option<&'a Path>,
}

/// Supplies bearer tokens for servers using external OAuth (e.g. Gmail).
#[async_trait]
pub trait McpOAuthProvider: Send + Sync {
    async fn access_token(&self, server: &McpServerEntry) -> Result<Option<String>, McpError>;
}

pub async fn connect_enabled_servers(options: McpConnectOptions<'_>) -> McpConnectSummary {
    let mut sessions = Vec::new();
    let mut connected_ids = Vec::new();

    for server in enabled_servers(options.config) {
        match connect_server(
            &server,
            options.full_registry,
            options.readonly_registry,
            options.oauth,
        )
        .await
        {
            Ok(conn) => {
                if let Some(path) = options.status_path {
                    let _ = record_server_status(
                        path,
                        &server.id,
                        &McpServerStatus {
                            last_connect_at: Some(Utc::now().to_rfc3339()),
                            last_connect_ok: Some(true),
                            last_connect_message: Some(format!("{} tools", conn.tool_count)),
                            last_error: None,
                        },
                    );
                }
                connected_ids.push(server.id.clone());
                sessions.extend(conn.sessions);
            }
            Err(e) => {
                if let Some(path) = options.status_path {
                    let _ = record_server_status(
                        path,
                        &server.id,
                        &McpServerStatus {
                            last_connect_at: Some(Utc::now().to_rfc3339()),
                            last_connect_ok: Some(false),
                            last_connect_message: None,
                            last_error: Some(e.to_string()),
                        },
                    );
                }
            }
        }
    }

    McpConnectSummary {
        sessions,
        connected_ids,
    }
}

async fn connect_server(
    server: &McpServerEntry,
    full_registry: &ToolRegistry,
    readonly_registry: &ToolRegistry,
    oauth: Option<&dyn McpOAuthProvider>,
) -> Result<McpConnections, McpError> {
    let custom_headers = resolve_headers(&server.headers);
    let auth_header = resolve_auth_header(server, oauth).await?;

    let session = McpSession::connect_http(McpHttpConfig {
        url: server.url.clone(),
        label: Some(server.id.clone()),
        auth_header,
        custom_headers,
    })
    .await?;

    let prefix = server.id.as_str();
    let tool_count = if server.tools.uses_split_registration() {
        register_split_tools(&session, full_registry, readonly_registry, prefix, &server.tools)
            .await?
    } else {
        let readonly = session.register_tools(readonly_registry, Some(prefix)).await?;
        session.register_tools(full_registry, Some(prefix)).await?;
        readonly.len()
    };

    Ok(McpConnections {
        sessions: vec![session],
        tool_count,
    })
}

async fn resolve_auth_header(
    server: &McpServerEntry,
    oauth: Option<&dyn McpOAuthProvider>,
) -> Result<Option<String>, McpError> {
    match server.auth {
        McpAuthKind::None => Ok(None),
        McpAuthKind::BearerEnv => Ok(server
            .bearer_env_var
            .as_deref()
            .and_then(|var| std::env::var(var).ok())
            .filter(|s| !s.is_empty())),
        McpAuthKind::GmailOAuth => {
            let Some(provider) = oauth else {
                return Err(McpError::Connect(format!(
                    "{} requires an OAuth provider",
                    server.id
                )));
            };
            provider.access_token(server).await
        }
    }
}

async fn register_split_tools(
    session: &Arc<McpSession>,
    full_registry: &ToolRegistry,
    readonly_registry: &ToolRegistry,
    prefix: &str,
    tools: &McpToolPolicy,
) -> Result<usize, McpError> {
    let readonly_set: HashMap<&str, ()> = tools.readonly.iter().map(|s| s.as_str()).map(|s| (s, ())).collect();
    let write_set: HashMap<&str, ()> = tools.write.iter().map(|s| s.as_str()).map(|s| (s, ())).collect();

    let readonly = session
        .register_tools_if(readonly_registry, Some(prefix), |name| {
            readonly_set.contains_key(name)
        })
        .await?;

    session
        .register_tools_if(full_registry, Some(prefix), |name| write_set.contains_key(name))
        .await?;

    session
        .register_tools_if(full_registry, Some(prefix), |name| readonly_set.contains_key(name))
        .await?;

    Ok(readonly.len() + write_set.len())
}
