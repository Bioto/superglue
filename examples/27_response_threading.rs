//! Example 27: Responses API tool loop with `previous_response_id` threading.

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};
use support::client_from_env;

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: Some("Echo JSON arguments".into()),
            parameters_schema: json!({"type": "object"}),
        }
    }

    async fn call(&self, arguments: serde_json::Value) -> Result<serde_json::Value, ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;
    client.register_tool(Arc::new(EchoTool)).await?;

    let outcome = client
        .complete_response(
            "Call the echo tool with {\"n\": 7} then summarize the echo result in one short sentence.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("rounds: {}", outcome.rounds);
    println!("response id: {}", outcome.id);
    Ok(())
}
