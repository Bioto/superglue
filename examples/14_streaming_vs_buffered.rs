//! Example 14: Compare buffered `complete` vs streaming latency perception.

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = client_from_env()?;
    let prompt = "Explain photosynthesis in two short sentences.";

    let buffered_start = std::time::Instant::now();
    let buffered = client
        .complete(prompt, superglue::CallOptions::default())
        .await?;
    let buffered_elapsed = buffered_start.elapsed();

    print!("stream: ");
    let stream_start = std::time::Instant::now();
    let mut first_token = None;
    let streamed = client
        .stream(prompt, superglue::CallOptions::default(), |_| {
            if first_token.is_none() {
                first_token = Some(std::time::Instant::now());
            }
        })
        .await?;
    let stream_elapsed = stream_start.elapsed();

    println!();
    println!("buffered content: {:?}", buffered.content);
    println!("buffered elapsed: {:.2?}", buffered_elapsed);
    if let Some(ft) = first_token {
        println!("stream first token: {:.2?} after start", ft - stream_start);
    }
    println!("stream total elapsed: {:.2?}", stream_elapsed);
    println!("stream content len: {} chars", streamed.content.len());
    Ok(())
}
