//! Connect MCP servers from [`McpServersFile`] configuration.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use chrono::Utc;

use crate::tools::ToolRegistry;

use super::config::{
    McpAuthKind, McpServerEntry, McpServerStatus, McpServersFile, McpToolPolicy, enabled_servers,
    record_server_status, resolve_headers,
};
use super::{McpError, McpHttpConfig, McpSession};

pub struct McpConnections {
    pub sessions: Vec<Arc<McpSession>>,
    pub tool_count: usize,
    pub registered_names: Vec<String>,
}

pub struct McpConnectSummary {
    pub sessions: Vec<Arc<McpSession>>,
    pub connected_ids: Vec<String>,
    /// All tool names registered in both registries during this connect.
    pub registered_tool_names: Vec<String>,
}

pub struct McpConnectOptions<'a> {
    pub config: &'a McpServersFile,
    pub full_registry: &'a ToolRegistry,
    pub readonly_registry: &'a ToolRegistry,
    pub status_path: Option<&'a Path>,
}

pub async fn connect_enabled_servers(options: McpConnectOptions<'_>) -> McpConnectSummary {
    let mut sessions = Vec::new();
    let mut connected_ids = Vec::new();
    let mut registered_tool_names = Vec::new();

    for server in enabled_servers(options.config) {
        match connect_server(&server, options.full_registry, options.readonly_registry).await {
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
                registered_tool_names.extend(conn.registered_names);
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
        registered_tool_names,
    }
}

async fn connect_server(
    server: &McpServerEntry,
    full_registry: &ToolRegistry,
    readonly_registry: &ToolRegistry,
) -> Result<McpConnections, McpError> {
    let custom_headers = resolve_headers(&server.headers);
    let auth_header = resolve_auth_header(server)?;

    let session = McpSession::connect_http(McpHttpConfig {
        url: server.url.clone(),
        label: Some(server.id.clone()),
        auth_header,
        custom_headers,
    })
    .await?;

    let prefix = server.id.as_str();
    let registered_names = if server.tools.uses_split_registration() {
        register_split_tools(
            &session,
            full_registry,
            readonly_registry,
            prefix,
            &server.tools,
        )
        .await?
    } else {
        let readonly_names = session
            .register_tools(readonly_registry, Some(prefix))
            .await?;
        let mut full_names = session.register_tools(full_registry, Some(prefix)).await?;
        // Merge: readonly names + any additional names registered only in full.
        let mut names = readonly_names;
        for n in full_names.drain(..) {
            if !names.contains(&n) {
                names.push(n);
            }
        }
        names
    };

    Ok(McpConnections {
        sessions: vec![session],
        tool_count: registered_names.len(),
        registered_names,
    })
}

fn resolve_auth_header(server: &McpServerEntry) -> Result<Option<String>, McpError> {
    match server.auth {
        McpAuthKind::None => Ok(None),
        McpAuthKind::BearerEnv => Ok(server
            .bearer_env_var
            .as_deref()
            .and_then(|var| std::env::var(var).ok())
            .filter(|s| !s.is_empty())),
    }
}

async fn register_split_tools(
    session: &Arc<McpSession>,
    full_registry: &ToolRegistry,
    readonly_registry: &ToolRegistry,
    prefix: &str,
    tools: &McpToolPolicy,
) -> Result<Vec<String>, McpError> {
    let readonly_set: HashMap<&str, ()> = tools
        .readonly
        .iter()
        .map(|s| s.as_str())
        .map(|s| (s, ()))
        .collect();
    let write_set: HashMap<&str, ()> = tools
        .write
        .iter()
        .map(|s| s.as_str())
        .map(|s| (s, ()))
        .collect();

    let mut all_names = session
        .register_tools_if(readonly_registry, Some(prefix), |name| {
            readonly_set.contains_key(name)
        })
        .await?;

    let write_names = session
        .register_tools_if(full_registry, Some(prefix), |name| {
            write_set.contains_key(name)
        })
        .await?;

    let readonly_in_full = session
        .register_tools_if(full_registry, Some(prefix), |name| {
            readonly_set.contains_key(name)
        })
        .await?;

    for n in write_names.into_iter().chain(readonly_in_full) {
        if !all_names.contains(&n) {
            all_names.push(n);
        }
    }

    Ok(all_names)
}
