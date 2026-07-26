//! Example 32: GlueLLM-style context optimization — dynamic routing, condensing, static tools.
//!
//! Requires `OPENAI_API_KEY`. Uses `tool_mode=Dynamic` so the model first calls `request_tools`,
//! then receives only the matched tool schemas. `condense_tool_messages` collapses each tool
//! round into a single summary message.

mod support;

use superglue::context::SummarizeContextConfig;
use superglue::tools::ToolMode;
use support::context_tools::register_context_optimization_tools;
use support::usage_line;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(support::require_api_key())
        .model(support::model())
        .base_url(support::base_url())
        .system_prompt("You are a helpful assistant. Use tools when needed.")
        .tool_mode(ToolMode::Dynamic)
        .condense_tool_messages(true)
        .summarize_context(SummarizeContextConfig {
            enabled: false,
            threshold: 20,
            keep_recent: 6,
        })
        .build()?;

    register_context_optimization_tools(&client).await?;

    let outcome = client
        .complete(
            "What's the weather in Paris?",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("{}", usage_line(&outcome));
    println!("rounds: {}", outcome.rounds);

    let condensed = outcome.messages.iter().any(|m| {
        m.content
            .as_ref()
            .and_then(|c| c.as_text())
            .is_some_and(|t| t.contains("[Tool Results]"))
    });
    println!("condensed tool round in history: {condensed}");

    Ok(())
}
