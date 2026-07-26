//! Example 16: Retry configuration and backoff schedules.

mod support;

use superglue::http::RetryPolicy;
use support::model;

fn backoff_schedule(initial_ms: u64, multiplier: f64, max_ms: u64, retries: u32) -> Vec<u64> {
    let mut delays = Vec::new();
    for i in 0..retries {
        let d = (initial_ms as f64 * multiplier.powi(i as i32)).min(max_ms as f64) as u64;
        delays.push(d);
    }
    delays
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let configs: [(&str, u32, u64, u64, f64); 4] = [
        ("Default", 3, 50, 2000, 2.0),
        ("Aggressive", 6, 100, 30_000, 2.0),
        ("Impatient", 1, 200, 200, 1.0),
        ("No retries", 0, 0, 0, 1.0),
    ];

    println!("Backoff schedules (ms between each attempt):\n");
    println!("Default connection settings: connect_timeout=30s, pool_max_idle_per_host=50\n");

    for (name, retries, initial, max_d, mult) in configs {
        let schedule = if retries > 0 {
            backoff_schedule(initial, mult, max_d, retries)
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(" → ")
        } else {
            "—".to_string()
        };
        println!("  {:12} ({retries} retries): {schedule}", name);
    }
    println!();

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        println!("Set OPENAI_API_KEY to run a live completion with the production config.");
        return Ok(());
    }

    let client = superglue::Client::builder()
        .api_key(api_key)
        .model(model())
        .max_retries(5)
        .retry_initial_delay_ms(100)
        .retry_max_delay_ms(10_000)
        .retry_multiplier(2.0)
        .requests_per_second(8)
        .timeout(std::time::Duration::from_secs(120))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    let _ = RetryPolicy::default();

    let outcome = client
        .complete(
            "In one sentence, why is exponential backoff better than fixed-interval retry?",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("Response: {:?}", outcome.content);
    if let Some(u) = &outcome.usage {
        println!(
            "Usage: prompt={} completion={} total={}",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        );
    }
    Ok(())
}
