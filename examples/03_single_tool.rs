//! Example 03: Single tool — model decides when to invoke it.

mod support;

use serde_json::json;
use superglue::tools::ToolSpec;
use support::{FnTool, usage_line};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(support::require_api_key())
        .model(support::model())
        .base_url(support::base_url())
        .system_prompt("You are a helpful weather assistant. Use tools to answer questions.")
        .build()?;

    let weather = FnTool::new(
        ToolSpec {
            name: "get_weather".into(),
            description: Some("Get the current weather for a given location.".into()),
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "City name" },
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                },
                "required": ["location"]
            }),
            static_tool: false,
        },
        |args| {
            let location = args
                .get("location")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let unit = args
                .get("unit")
                .and_then(|v| v.as_str())
                .unwrap_or("celsius");
            Ok(json!({
                "location": location,
                "temperature": 22,
                "unit": unit,
                "condition": "sunny"
            }))
        },
    );
    client.register_tool(weather.into_arc()).await?;

    let outcome = client
        .complete(
            "What's the weather like in Tokyo right now?",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("rounds: {}", outcome.rounds);
    println!("{}", usage_line(&outcome));
    Ok(())
}
