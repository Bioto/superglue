//! Example 04: Multiple tools registered on one client.

mod support;

use serde_json::json;
use support::{FnTool, usage_line};
use superglue::tools::ToolSpec;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(support::require_api_key())
        .model(support::model())
        .base_url(support::base_url())
        .system_prompt("Use the available tools when needed.")
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "add".into(),
                    description: Some("Add two integers.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": {
                            "a": { "type": "integer" },
                            "b": { "type": "integer" }
                        },
                        "required": ["a", "b"]
                    }),
                },
                |args| {
                    let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({ "result": a + b }))
                },
            )
            .into_arc(),
        )
        .await?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "multiply".into(),
                    description: Some("Multiply two integers.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": {
                            "a": { "type": "integer" },
                            "b": { "type": "integer" }
                        },
                        "required": ["a", "b"]
                    }),
                },
                |args| {
                    let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({ "result": a * b }))
                },
            )
            .into_arc(),
        )
        .await?;

    let outcome = client
        .complete(
            "What is (3 + 4) multiplied by 5? Use the tools.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("rounds: {}", outcome.rounds);
    println!("{}", usage_line(&outcome));
    Ok(())
}
