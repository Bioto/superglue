//! Example 25: Model fallback chain — primary model with backup list.

mod support;

use superglue::{Client, CallOptions};
use support::{model, require_api_key, base_url};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let primary = model();
    let backup = std::env::var("OPENAI_FALLBACK_MODEL").unwrap_or_else(|_| primary.clone());
    let client = Client::builder()
        .api_key(require_api_key())
        .model(primary.clone())
        .base_url(base_url())
        .model_fallback_chain(vec![primary.clone(), backup], None)
        .build()?;

    let outcome = client
        .complete("Reply with exactly: fallback ok", CallOptions::default())
        .await?;

    println!("content: {:?}", outcome.content);
    println!("model_used: {}", outcome.model_used);
    Ok(())
}
