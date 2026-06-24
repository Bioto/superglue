//! JSON configuration for HTTP MCP server connections.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_GMAIL_MCP_URL: &str = "https://gmailmcp.googleapis.com/mcp/v1";
pub const DEFAULT_CONTEXT7_MCP_URL: &str = "https://mcp.context7.com/mcp";

#[derive(Debug, Error)]
pub enum McpConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServersFile {
    #[serde(default = "default_server_entries")]
    pub servers: Vec<McpServerEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerEntry {
    pub id: String,
    #[serde(default)]
    pub enabled: bool,
    pub url: String,
    #[serde(default)]
    pub auth: McpAuthKind,
    #[serde(default, rename = "bearer_env")]
    pub bearer_env_var: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub tools: McpToolPolicy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum McpAuthKind {
    #[default]
    None,
    #[serde(rename = "gmail_oauth")]
    GmailOAuth,
    BearerEnv,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct McpToolPolicy {
    #[serde(default)]
    pub readonly: Vec<String>,
    #[serde(default)]
    pub write: Vec<String>,
}

impl McpToolPolicy {
    #[must_use]
    pub fn uses_split_registration(&self) -> bool {
        !self.readonly.is_empty() || !self.write.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct McpServerStatus {
    #[serde(default)]
    pub last_connect_at: Option<String>,
    #[serde(default)]
    pub last_connect_ok: Option<bool>,
    #[serde(default)]
    pub last_connect_message: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct McpServersStatusFile {
    #[serde(default)]
    pub servers: HashMap<String, McpServerStatus>,
}

fn default_server_entries() -> Vec<McpServerEntry> {
    default_mcp_servers().servers
}

#[must_use]
pub fn default_mcp_servers() -> McpServersFile {
    McpServersFile {
        servers: vec![
            McpServerEntry {
                id: "gmail".into(),
                enabled: false,
                url: DEFAULT_GMAIL_MCP_URL.into(),
                auth: McpAuthKind::GmailOAuth,
                bearer_env_var: None,
                headers: HashMap::new(),
                tools: McpToolPolicy {
                    readonly: vec![
                        "search_threads".into(),
                        "get_thread".into(),
                        "list_drafts".into(),
                        "list_labels".into(),
                    ],
                    write: vec![
                        "create_draft".into(),
                        "label_message".into(),
                        "label_thread".into(),
                        "unlabel_message".into(),
                        "unlabel_thread".into(),
                        "create_label".into(),
                    ],
                },
            },
            McpServerEntry {
                id: "context7".into(),
                enabled: false,
                url: DEFAULT_CONTEXT7_MCP_URL.into(),
                auth: McpAuthKind::None,
                bearer_env_var: None,
                headers: HashMap::from([(
                    "CONTEXT7_API_KEY".into(),
                    "$CONTEXT7_API_KEY".into(),
                )]),
                tools: McpToolPolicy::default(),
            },
        ],
    }
}

pub fn load_servers_file(path: &Path) -> Result<McpServersFile, McpConfigError> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn save_servers_file(path: &Path, file: &McpServersFile) -> Result<(), McpConfigError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(file)?;
    fs::write(path, json)?;
    Ok(())
}

/// Write defaults when `path` does not exist.
pub fn ensure_servers_file(path: &Path) -> Result<(), McpConfigError> {
    if path.exists() {
        return Ok(());
    }
    save_servers_file(path, &default_mcp_servers())
}

pub fn load_servers_file_or_default(path: &Path) -> McpServersFile {
    load_servers_file(path)
        .ok()
        .unwrap_or_else(default_mcp_servers)
}

#[must_use]
pub fn enabled_servers(config: &McpServersFile) -> Vec<McpServerEntry> {
    config
        .servers
        .iter()
        .filter(|s| s.enabled)
        .cloned()
        .collect()
}

#[must_use]
pub fn resolve_env_value(value: &str) -> String {
    if let Some(var) = value.strip_prefix('$') {
        env::var(var).unwrap_or_default()
    } else {
        value.to_string()
    }
}

#[must_use]
pub fn resolve_headers(headers: &HashMap<String, String>) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let resolved = resolve_env_value(value);
            if resolved.is_empty() {
                None
            } else {
                Some((name.clone(), resolved))
            }
        })
        .collect()
}

pub fn load_status_file(path: &Path) -> McpServersStatusFile {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn save_status_file(path: &Path, file: &McpServersStatusFile) -> Result<(), McpConfigError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(file)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn record_server_status(
    path: &Path,
    server_id: &str,
    status: &McpServerStatus,
) -> Result<(), McpConfigError> {
    let mut file = load_status_file(path);
    file.servers.insert(server_id.to_string(), status.clone());
    save_status_file(path, &file)
}

pub fn format_config_status(config_path: &Path, status_path: &Path) -> String {
    let config = load_servers_file_or_default(config_path);
    let status = load_status_file(status_path);
    let mut lines = vec![format!("config: {}", config_path.display())];

    for server in &config.servers {
        let state = if server.enabled { "enabled" } else { "disabled" };
        lines.push(format!("  {} ({state}): {}", server.id, server.url));
        if let Some(entry) = status.servers.get(&server.id) {
            if let Some(ok) = entry.last_connect_ok {
                let msg = entry
                    .last_connect_message
                    .as_deref()
                    .unwrap_or(if ok { "ok" } else { "failed" });
                lines.push(format!(
                    "    last connect: {} ({msg})",
                    if ok { "ok" } else { "failed" }
                ));
            }
            if let Some(err) = entry.last_error.as_deref().filter(|s| !s.is_empty()) {
                lines.push(format!("    last error: {err}"));
            }
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn resolve_env_value_expands_dollar_prefix() {
        unsafe {
            env::set_var("SG_TEST_MCP_KEY", "secret");
        }
        assert_eq!(resolve_env_value("$SG_TEST_MCP_KEY"), "secret");
        assert_eq!(resolve_env_value("literal"), "literal");
        unsafe {
            env::remove_var("SG_TEST_MCP_KEY");
        }
    }

    #[test]
    fn ensure_servers_file_writes_defaults() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("servers.json");
        ensure_servers_file(&path).unwrap();
        let loaded = load_servers_file(&path).unwrap();
        assert_eq!(loaded.servers.len(), 2);
    }

    #[test]
    fn enabled_servers_filters_disabled() {
        let mut file = default_mcp_servers();
        file.servers[0].enabled = true;
        let enabled = enabled_servers(&file);
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].id, "gmail");
    }
}
