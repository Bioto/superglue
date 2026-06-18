//! Demo: simple client + one tool.

mod support;

use serde_json::json;
use support::{FnTool, model};
use superglue::tools::ToolSpec;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("superglue.version(): {}", superglue::version());

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        println!(
            "\nOPENAI_API_KEY not set — demonstrating client construction only.\n\
             Set the env var and re-run for a live completion."
        );
        let client = superglue::Client::builder()
            .api_key("sk-placeholder")
            .model(model())
            .system_prompt("You are a concise assistant.")
            .build()?;

        client
            .register_tool(
                FnTool::new(
                    ToolSpec {
                        name: "get_weather".into(),
                        description: Some("Get weather for a city.".into()),
                        parameters_schema: json!({
                            "type": "object",
                            "properties": { "location": { "type": "string" } },
                            "required": ["location"]
                        }),
                    },
                    |args| {
                        let loc = args
                            .get("location")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        Ok(json!({ "location": loc, "temp": 22, "unit": "C" }))
                    },
                )
                .into_arc(),
            )
            .await?;

        println!("Client and tool registered successfully (no network call made).");
        return Ok(());
    }

    let client = superglue::Client::builder()
        .api_key(api_key)
        .model(model())
        .base_url(support::base_url())
        .system_prompt("You are a concise assistant. Use tools when relevant.")
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "get_weather".into(),
                    description: Some("Get weather for a city.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": { "location": { "type": "string" } },
                        "required": ["location"]
                    }),
                },
                |args| {
                    let loc = args
                        .get("location")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    Ok(json!({ "location": loc, "temp": 22, "unit": "C" }))
                },
            )
            .into_arc(),
        )
        .await?;

    let outcome = client
        .complete("What is the weather in Paris?", superglue::CallOptions::default())
        .await?;

    println!("\ncontent: {:?}", outcome.content);
    println!("rounds: {}", outcome.rounds);
    if let Some(u) = &outcome.usage {
        println!(
            "usage: prompt={} completion={} total={}",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        );
    }
    Ok(())
}
