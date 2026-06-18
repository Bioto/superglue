//! Example 01: Simple text completion with no tools.

mod support;

use support::{client_from_env, usage_line};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;
    let outcome = client
        .complete("What is the capital of France? Reply in one sentence.", superglue::CallOptions::default())
        .await?;
    println!("content: {:?}", outcome.content);
    println!("rounds: {}", outcome.rounds);
    println!("{}", usage_line(&outcome));
    Ok(())
}
