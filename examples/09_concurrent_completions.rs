//! Example 09: Concurrent completions with `tokio::join`.

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;

    let prompts = [
        "Name the tallest mountain on Earth.",
        "What year did the Berlin Wall fall?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the speed of light in m/s?",
        "Name the first programming language.",
    ];

    let start = std::time::Instant::now();

    let futures: Vec<_> = prompts
        .iter()
        .map(|p| {
            let c = client.clone();
            let prompt = *p;
            async move {
                let outcome = c
                    .complete(prompt, superglue::CallOptions::default())
                    .await?;
                Ok((prompt, outcome.content))
            }
        })
        .collect();

    let results: Vec<Result<(&str, Option<String>), superglue::chat::ChatError>> =
        futures_util::future::join_all(futures).await;

    for r in results {
        let (prompt, answer) = r?;
        println!("Q: {prompt}");
        println!("A: {:?}\n", answer);
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "All {} completions finished in {:.2}s (concurrent).",
        prompts.len(),
        elapsed
    );
    Ok(())
}
