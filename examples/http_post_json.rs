//! Demo: [`superglue::http::HttpClient::post_json`] against a public echo endpoint (needs network).
//!
//! Run: `cargo run --example http_post_json`
//!
//! Uses <https://httpbin.org/post> which returns JSON including the posted body.

use serde_json::json;

use superglue::http::{ClientConfig, HttpClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = HttpClient::new(ClientConfig::default())?;
    let url = "https://httpbin.org/post";
    let body = json!({
        "example": "http_post_json",
        "n": 1
    });
    let value = client.post_json(url, &body).await?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
