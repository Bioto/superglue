//! Example 20: Agent system (`AgentSpec`, `AgentEngine`, `Client::run_agent`).

mod support;

use serde_json::json;
use support::{model, require_api_key, FnTool};
use superglue::agents::{AgentEngine, AgentSpec};
use superglue::tools::ToolSpec;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = require_api_key();
    let model = model();

    println!("{}", "=".repeat(60));
    println!("Example 1: Minimal agent (persona only)");
    println!("{}", "=".repeat(60));

    let spec = AgentSpec::new("SimpleHelper", "a concise and helpful assistant");
    println!("Compiled system prompt:");
    println!("{}", spec.compile_system_prompt());
    println!();

    let client = superglue::Client::builder()
        .api_key(api_key.clone())
        .model(model.clone())
        .build()?;

    let outcome = client
        .run_agent(spec, "What is 2 + 2? Answer in one sentence.", superglue::CallOptions::default())
        .await?;
    println!("Response: {:?}", outcome.content);
    println!();

    println!("{}", "=".repeat(60));
    println!("Example 2: Agent with goals and constraints");
    println!("{}", "=".repeat(60));

    let research_spec = AgentSpec::new(
        "ResearchAssistant",
        "an expert research assistant specialising in science and technology",
    )
        .with_goal("Provide accurate, well-sourced answers backed by evidence.")
        .with_goal("Explain complex concepts clearly for a general audience.")
        .with_constraint("Never speculate without clearly labelling it as speculation.")
        .with_constraint("Keep responses concise — prefer bullet points over long paragraphs.")
        .with_model(model.clone())
        .with_max_tool_rounds(4);

    println!("Compiled system prompt:");
    println!("{}", research_spec.compile_system_prompt());
    println!();

    let outcome = client
        .run_agent(
            research_spec,
            "Explain quantum entanglement in 3 bullet points.",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("Response: {:?}", outcome.content);
    println!();

    println!("{}", "=".repeat(60));
    println!("Example 3: Agent with tools via AgentEngine");
    println!("{}", "=".repeat(60));

    let agent_spec = AgentSpec::new("WeatherAgent", "a friendly meteorologist")
        .with_goal("Provide accurate weather data for any city asked.")
        .with_model(model.clone());

    let http_client =
        superglue::http::HttpClient::new(superglue::http::ClientConfig::default())?;
    let registry = superglue::tools::ToolRegistry::new();
    registry
        .register(
            FnTool::new(
                ToolSpec {
                    name: "get_temperature".into(),
                    description: Some("Return temperature for a city.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }),
                },
                |args| {
                    let city = args
                        .get("city")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_lowercase();
                    let temp = match city.as_str() {
                        "london" => 14,
                        "tokyo" => 22,
                        _ => 20,
                    };
                    Ok(json!({ "city": city, "temperature_celsius": temp }))
                },
            )
            .into_arc(),
        )
        .await?;

    let engine = AgentEngine::new(agent_spec);
    let opts = superglue::chat::ChatOptions::new(
        support::base_url(),
        api_key.clone(),
        model.clone(),
    );
    let outcome = engine
        .run(
            &http_client,
            &registry,
            "What is the current temperature in Tokyo?",
            &opts,
        )
        .await?;
    println!("Response: {:?}", outcome.content);
    println!();

    println!("{}", "=".repeat(60));
    println!("Example 4: reasoning_effort on agent");
    println!("{}", "=".repeat(60));

    let mut reasoning_spec = AgentSpec::new("DeepThinker", "a careful analyst")
        .with_model(model);
    reasoning_spec.reasoning_effort = Some("high".to_string());

    let outcome = client
        .run_agent(
            reasoning_spec,
            "In one sentence, why use exponential backoff?",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("Response: {:?}", outcome.content);

    let _ = engine;
    Ok(())
}
