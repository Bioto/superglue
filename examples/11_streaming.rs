//! Example 11: Streaming tokens to stdout.

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;

    print!("STREAM: ");
    let outcome = client
        .stream(
            "Count from 1 to 5 with commas.",
            superglue::CallOptions::default(),
            |delta| {
                print!("{delta}");
            },
        )
        .await?;

    println!();
    println!(
        "done finish_reason={:?} request_id={}",
        outcome.finish_reason, outcome.request_id
    );
    Ok(())
}
