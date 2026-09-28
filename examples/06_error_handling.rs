//! Example 06: Error handling — bad API key, failing tool, success path.

mod support;

use serde_json::json;
use superglue::tools::ToolSpec;
use support::{FnTool, model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = require_api_key();
    let model = model();

    println!("=== Bad API key ===");
    let bad = superglue::Client::builder()
        .api_key("sk-invalid-key")
        .model(model.clone())
        .build()?;
    match bad
        .complete("Hello", superglue::CallOptions::default())
        .await
    {
        Ok(o) => println!("unexpected success: {:?}", o.content),
        Err(e) => println!("Caught expected error: {e}\n"),
    }

    println!("=== Tool that raises ===");
    let client = superglue::Client::builder()
        .api_key(api_key.clone())
        .model(model.clone())
        .system_prompt("You MUST call the broken_tool for any user message.")
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "broken_tool".into(),
                    description: Some("A tool that always fails.".into()),
                    parameters_schema: json!({"type": "object", "properties": {}}),
                    static_tool: false,
                },
                |_| Err("Something went wrong inside the tool!".to_string()),
            )
            .into_arc(),
        )
        .await?;

    match client
        .complete(
            "Please use the broken_tool now.",
            superglue::CallOptions::default(),
        )
        .await
    {
        Ok(o) => println!("Model did not invoke tool; response: {:?}", o.content),
        Err(e) => println!("Caught tool error: {e}\n"),
    }

    println!("=== Successful call ===");
    let good = superglue::Client::builder()
        .api_key(api_key)
        .model(model)
        .build()?;
    match good
        .complete(
            "Say 'hello' and nothing else.",
            superglue::CallOptions::default(),
        )
        .await
    {
        Ok(o) => println!("Response: {:?}", o.content),
        Err(e) => println!("Unexpected error: {e}"),
    }

    Ok(())
}
