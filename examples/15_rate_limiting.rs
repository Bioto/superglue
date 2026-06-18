//! Example 15: Rate limiting via `requests_per_second`.

mod support;

use support::{model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .requests_per_second(2)
        .build()?;

    let questions = [
        "Name the planet closest to the Sun. One word.",
        "What colour is the sky on a clear day? One word.",
        "How many days are in a week? One word.",
        "What is the chemical symbol for water? One word.",
        "Name the largest ocean on Earth. Two words max.",
        "What is 7 multiplied by 8? One number.",
    ];

    println!("Sending {} requests with requests_per_second=2 …\n", questions.len());
    let start = std::time::Instant::now();

    for (i, question) in questions.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let outcome = client
            .complete(*question, superglue::CallOptions::default())
            .await?;
        let elapsed = t0.elapsed().as_secs_f64();
        let total = start.elapsed().as_secs_f64();
        println!(
            "[{}/{}] t={total:.2}s (+{elapsed:.2}s) → {:?}",
            i + 1,
            questions.len(),
            outcome.content
        );
    }

    let total_elapsed = start.elapsed().as_secs_f64();
    println!("\nAll done in {total_elapsed:.2}s.");
    println!(
        "With a 2 req/s cap, 6 requests need at least {:.1}s — limiter working if total ≥ that.",
        (questions.len() as f64 - 1.0) / 2.0
    );
    Ok(())
}
