//! Example 29: same prompt across provider:model strings (wiremock-friendly when base_url overridden).

use superglue::client::{CallOptions, ClientBuilder};
use superglue::providers::ProviderId;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com".to_string());

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if openai_key.is_empty() {
        eprintln!("Set OPENAI_API_KEY to run this example.");
        return Ok(());
    }

    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    let mut models = vec!["openai:gpt-4o-mini"];
    if anthropic_key.is_empty() {
        eprintln!("ANTHROPIC_API_KEY not set — skipping anthropic model.");
    } else {
        models.push("anthropic:claude-sonnet-4-20250514");
    }

    for model in models {
        let client = ClientBuilder::new()
            .api_key(openai_key.clone())
            .api_key_for(ProviderId::OpenAi, openai_key.clone())
            .api_key_for(ProviderId::Anthropic, anthropic_key.clone())
            .base_url(&base)
            .model(model)
            .build()?;
        let out = client
            .complete("Say hello in one word.", CallOptions::default())
            .await?;
        println!("{model}: {}", out.content.unwrap_or_default());
    }
    Ok(())
}
