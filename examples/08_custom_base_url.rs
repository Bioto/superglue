//! Example 08: Custom `base_url` for OpenAI-compatible providers (Ollama, etc.).

mod support;

use support::{base_url, model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = require_api_key();
    let url = base_url();
    let m = model();

    println!("Using base_url={url} model={m}");

    let client = superglue::Client::builder()
        .api_key(key)
        .model(m)
        .base_url(url)
        .build()?;

    let outcome = client
        .complete(
            "Reply with exactly one word: ok.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    Ok(())
}
