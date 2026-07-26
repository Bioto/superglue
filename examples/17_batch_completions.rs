//! Example 17: Batch completions.

mod support;

use superglue::batch::{BatchConfig, BatchRequest};
use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;

    let requests = vec![
        BatchRequest::new("Say: one."),
        BatchRequest::new("Say: two."),
        BatchRequest::new("Say: three."),
    ];

    let response = client
        .batch(
            requests,
            BatchConfig {
                max_concurrent: 2,
                ..BatchConfig::default()
            },
        )
        .await?;

    println!(
        "batch ok={} fail={} elapsed={:.2}s",
        response.successful, response.failed, response.elapsed_secs
    );

    for r in &response.results {
        println!(
            "{} success={} content={:?}",
            r.id,
            r.success,
            r.content
                .as_deref()
                .map(|c| c.chars().take(80).collect::<String>())
        );
    }

    Ok(())
}
