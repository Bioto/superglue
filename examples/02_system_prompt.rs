//! Example 02: System prompt shapes assistant behaviour.

mod support;

use support::{model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .system_prompt("You are a pirate. Always respond in pirate speak.")
        .build()?;

    let outcome = client
        .complete(
            "What is the capital of France?",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("content: {:?}", outcome.content);
    Ok(())
}
