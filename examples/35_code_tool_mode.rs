//! Example 35: programmatic tool calling (`tool_mode=Code`).
//!
//! The model first calls `request_tools`, receives matched tool schemas as JSON,
//! then writes JavaScript for the singular `code` tool.
//!
//! Requires `OPENAI_API_KEY`.

mod support;

use superglue::tools::ToolMode;
use support::context_tools::register_context_optimization_tools;
use support::usage_line;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(support::require_api_key())
        .model(support::model())
        .base_url(support::base_url())
        .system_prompt(
            "You are a helpful assistant. Call request_tools first, then use the code \
             tool to invoke matched tools from JavaScript (tools.<name>(args)).",
        )
        .tool_mode(ToolMode::Code)
        .build()?;

    register_context_optimization_tools(&client).await?;

    let outcome = client
        .complete(
            "What's the weather in Paris? Return only the temperature.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("{}", usage_line(&outcome));
    println!("rounds: {}", outcome.rounds);

    let used_code = outcome.messages.iter().any(|m| {
        m.name.as_deref() == Some("code")
            || m.tool_calls
                .as_ref()
                .is_some_and(|calls| calls.iter().any(|c| c.function.name == "code"))
    });
    println!("used code tool: {used_code}");

    Ok(())
}
