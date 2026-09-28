//! Example 12: Streaming with post-processing on accumulated content.

mod support;

use std::sync::atomic::{AtomicU32, Ordering};

use support::{model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    static TOKEN_COUNT: AtomicU32 = AtomicU32::new(0);

    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .system_prompt(
            "You are a concise assistant. Always include at least one number in your responses.",
        )
        .build()?;

    println!("Streaming:\n");
    let outcome = client
        .stream(
            "List 5 interesting facts about the planet Mars. Use a numbered list.",
            superglue::CallOptions::default(),
            |_| {
                TOKEN_COUNT.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await?;

    println!("\n--- stream ended ---\n");
    println!("content: {}", outcome.content);
    let numbers: Vec<_> = outcome
        .content
        .split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .collect();
    println!("Number-like tokens: {:?}", numbers);
    println!(
        "Callback fired: {} times",
        TOKEN_COUNT.load(Ordering::SeqCst)
    );
    println!("finish_reason: {:?}", outcome.finish_reason);
    if let Some(u) = &outcome.usage {
        println!(
            "usage: prompt={} completion={} total={}",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        );
    }
    Ok(())
}
