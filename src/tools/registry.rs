//! Async tool registry and dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::instrument;

use super::error::ToolInvokeError;
use super::types::{ToolRetryPolicy, ToolSpec};

/// Async tool implementation (JSON args in, JSON value out).
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError>;
}

/// Registry of tools keyed by name, each with an optional per-tool error policy.
#[derive(Default)]
pub struct ToolRegistry {
    tools: RwLock<HashMap<String, (Arc<dyn Tool>, ToolRetryPolicy)>>,
}

impl ToolRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tool with the default (fail-fast) error policy.
    ///
    /// Returns an error if the tool name is already registered.
    pub async fn register(&self, tool: Arc<dyn Tool>) -> Result<(), ToolInvokeError> {
        self.register_with_policy(tool, ToolRetryPolicy::default())
            .await
    }

    /// Register a tool with an explicit per-tool error policy.
    ///
    /// Returns an error if the tool name is already registered.
    pub async fn register_with_policy(
        &self,
        tool: Arc<dyn Tool>,
        policy: ToolRetryPolicy,
    ) -> Result<(), ToolInvokeError> {
        let name = tool.spec().name.clone();
        let mut map = self.tools.write().await;
        if map.contains_key(&name) {
            return Err(ToolInvokeError::DuplicateRegistration { name });
        }
        map.insert(name, (tool, policy));
        Ok(())
    }

    /// List registered tool specs (order not guaranteed).
    pub async fn list_specs(&self) -> Vec<ToolSpec> {
        let map = self.tools.read().await;
        map.values().map(|(t, _)| t.spec()).collect()
    }

    /// Return the error policy registered for `name`, or `ToolRetryPolicy::default()`.
    pub async fn policy_for(&self, name: &str) -> ToolRetryPolicy {
        let map = self.tools.read().await;
        map.get(name).map(|(_, p)| p.clone()).unwrap_or_default()
    }

    /// Unregister tools by name. Returns the number of tools actually removed.
    pub async fn unregister(&self, names: &[String]) -> usize {
        let mut map = self.tools.write().await;
        let mut removed = 0;
        for name in names {
            if map.remove(name).is_some() {
                removed += 1;
            }
        }
        removed
    }

    /// Invoke a tool by name.
    #[instrument(skip(self, arguments), fields(tool = name))]
    pub async fn invoke(&self, name: &str, arguments: Value) -> Result<Value, ToolInvokeError> {
        let tool = {
            let map = self.tools.read().await;
            map.get(name).map(|(t, _)| Arc::clone(t))
        };
        let Some(tool) = tool else {
            return Err(ToolInvokeError::UnknownTool {
                name: name.to_string(),
            });
        };
        tool.call(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: self.name.clone(),
                description: Some("echo".into()),
                parameters_schema: serde_json::json!({"type": "object", "properties": {}}),
                static_tool: false,
            }
        }

        async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
            Ok(arguments)
        }
    }

    fn make_tool(name: &str) -> Arc<dyn Tool> {
        Arc::new(EchoTool { name: name.into() })
    }

    #[tokio::test]
    async fn unregister_removes_tools() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("foo")).await.unwrap();
        registry.register(make_tool("bar")).await.unwrap();

        let removed = registry
            .unregister(&["foo".into(), "bar".into()])
            .await;
        assert_eq!(removed, 2);

        let specs = registry.list_specs().await;
        assert!(specs.is_empty());
    }

    #[tokio::test]
    async fn unregister_invoke_returns_unknown_tool() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("foo")).await.unwrap();
        registry.unregister(&["foo".into()]).await;

        let result = registry.invoke("foo", json!({})).await;
        assert!(matches!(result, Err(ToolInvokeError::UnknownTool { name }) if name == "foo"));
    }

    #[tokio::test]
    async fn unregister_unknown_name_returns_zero() {
        let registry = ToolRegistry::new();
        let removed = registry.unregister(&["nonexistent".into()]).await;
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn unregister_partial_names() {
        let registry = ToolRegistry::new();
        registry.register(make_tool("a")).await.unwrap();
        registry.register(make_tool("b")).await.unwrap();

        let removed = registry.unregister(&["a".into(), "missing".into()]).await;
        assert_eq!(removed, 1);

        let specs = registry.list_specs().await;
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "b");
    }
}
