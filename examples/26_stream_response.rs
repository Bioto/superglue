//! Example 26: Stream text via the OpenAI Responses API.

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;
    let mut tokens = Vec::new();
    let outcome = client
        .stream_response(
            "Count from 1 to 5 separated by spaces.",
            superglue::CallOptions::default(),
            |d| tokens.push(d),
        )
        .await?;

    println!("deltas: {:?}", tokens);
    println!("content: {}", outcome.content);
    println!("id: {}", outcome.id);
    println!("model_used: {}", outcome.model_used);
    Ok(())
}
