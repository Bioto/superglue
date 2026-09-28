//! Example 05: Token usage on completion outcomes.

mod support;

use support::{client_from_env, usage_line};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;
    let outcome = client
        .complete(
            "List three primary colours in one line.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    println!("{}", usage_line(&outcome));
    if let Some(u) = &outcome.usage {
        println!(
            "detail: prompt_tokens={} completion_tokens={} total_tokens={}",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        );
    }
    Ok(())
}
