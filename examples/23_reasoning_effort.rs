//! Example 23: Per-call and client-level `reasoning_effort`.

mod support;

use superglue::agents::AgentSpec;
use support::{model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .reasoning_effort("medium")
        .system_prompt("Answer in one short sentence.")
        .build()?;

    println!("1. Client-level reasoning_effort=medium");
    let outcome = client
        .complete("What is 17 + 25?", superglue::CallOptions::default())
        .await?;
    println!("   {}\n", outcome.content.unwrap_or_default());

    println!("2. Per-call override reasoning_effort=high");
    let outcome = client
        .complete(
            "Name one benefit of higher reasoning effort.",
            superglue::CallOptions {
                reasoning_effort: Some("high".to_string()),
                ..Default::default()
            },
        )
        .await?;
    println!("   {}\n", outcome.content.unwrap_or_default());

    println!("3. AgentSpec.reasoning_effort=high");
    let spec = AgentSpec::new("DeepThinker", "a careful analyst")
        .with_model(model())
        .with_max_tool_rounds(8);
    let mut agent_spec = spec;
    agent_spec.reasoning_effort = Some("high".to_string());

    let outcome = client
        .run_agent(
            agent_spec,
            "Why might exponential backoff beat fixed-interval retry?",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("   {}", outcome.content.unwrap_or_default());

    Ok(())
}
