//! Offline demo: [`superglue::tools::run_plan`] with a tiny echo tool (no network).
//!
//! Run: `cargo run --example tools_harness`

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use superglue::tools::{PlanStep, Tool, ToolInvocation, ToolRegistry, ToolSpec, run_plan};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: None,
            parameters_schema: json!({"type": "object"}),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool)).await?;

    let steps = vec![
        PlanStep::ToolCall(ToolInvocation {
            tool_name: "echo".into(),
            arguments: json!({ "note": "harness example" }),
        }),
        PlanStep::Done {
            message: "done".into(),
        },
    ];

    let events = run_plan(&registry, &steps).await?;
    for ev in events {
        println!("{ev:?}");
    }
    Ok(())
}
