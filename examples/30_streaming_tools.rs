//! Example 30: streaming completion with a registered tool (multi-round SSE).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use superglue::client::{CallOptions, ClientBuilder};
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: Some("Echo JSON args".into()),
            parameters_schema: json!({"type": "object"}),
            static_tool: false,
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
        .model(std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "openai:gpt-4o-mini".into()))
        .build()?;
    client.register_tool(Arc::new(EchoTool)).await?;

    let mut printed = String::new();
    let out = client
        .stream(
            "Call echo with {\"msg\":\"hi\"} then say OK",
            CallOptions::default(),
            |d| {
                printed.push_str(&d);
                print!("{d}");
            },
        )
        .await?;
    println!(
        "\nfinish={:?} rounds_stream_content={printed}",
        out.finish_reason
    );
    Ok(())
}
