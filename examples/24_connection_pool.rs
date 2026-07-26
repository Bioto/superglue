//! Example 24: HTTP connection pool and timeout defaults.

mod support;

use std::time::Duration;

use superglue::http::ClientConfig;
use support::model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let defaults = ClientConfig::default();
    println!("ClientConfig defaults:");
    println!("  connect_timeout = {:?}", defaults.connect_timeout);
    println!(
        "  pool_max_idle_per_host = {}",
        defaults.pool_max_idle_per_host
    );
    println!("  timeout = {:?}", defaults.timeout);
    println!();

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("Set OPENAI_API_KEY to run the live section.");
        return Ok(());
    }

    let client = superglue::Client::builder()
        .api_key(api_key)
        .model(model())
        .base_url(support::base_url())
        .connect_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(50)
        .timeout(Duration::from_secs(90))
        .build()?;

    let outcome = client
        .complete(
            "Say 'pool ok' in two words.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("Chat completion: {}", outcome.content.unwrap_or_default());
    Ok(())
}
