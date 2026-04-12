//! Async tool registry and dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::instrument;

use super::error::ToolInvokeError;
use super::types::ToolSpec;

/// Async tool implementation (JSON args in, JSON value out).
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError>;
}

/// Registry of tools keyed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool; returns error if the name is already registered.
    pub async fn register(&self, tool: Arc<dyn Tool>) -> Result<(), ToolInvokeError> {
        let name = tool.spec().name.clone();
        let mut map = self.tools.write().await;
        if map.insert(name.clone(), tool).is_some() {
            return Err(ToolInvokeError::DuplicateRegistration { name });
        }
        Ok(())
    }

    /// List registered tool specs (order not guaranteed).
    pub async fn list_specs(&self) -> Vec<ToolSpec> {
        let map = self.tools.read().await;
        map.values().map(|t| t.spec()).collect()
    }

    /// Invoke a tool by name. v1 policy: handler errors propagate as [`ToolInvokeError::HandlerFailed`].
    #[instrument(skip(self, arguments), fields(tool = name))]
    pub async fn invoke(&self, name: &str, arguments: Value) -> Result<Value, ToolInvokeError> {
        let tool = {
            let map = self.tools.read().await;
            map.get(name).cloned()
        };
        let Some(tool) = tool else {
            return Err(ToolInvokeError::UnknownTool {
                name: name.to_string(),
            });
        };
        tool.call(arguments).await
    }
}
