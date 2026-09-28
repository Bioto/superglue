//! Example 13: Multiple streaming requests in sequence.

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;

    let prompts = [
        "Say hello in one word.",
        "Name a colour in one word.",
        "Name an animal in one word.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        print!("[{}] ", i + 1);
        let outcome = client
            .stream(*prompt, superglue::CallOptions::default(), |delta| {
                print!("{delta}");
            })
            .await?;
        println!(" ({:?})", outcome.finish_reason);
    }

    Ok(())
}
