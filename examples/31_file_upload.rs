//! Example 31: inline file attachment + completion.

use superglue::client::{CallOptions, ClientBuilder};
use superglue::files::FilePurpose;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClientBuilder::new()
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
        .model(
            std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "openai:gpt-4o-mini".into()),
        )
        .build()?;

    let sample = superglue::files::demo_pdf_bytes();
    let msg = superglue::Client::message_with_file_bytes(
        Some("Summarize this file in one sentence."),
        "sample.pdf",
        sample,
    );

    let out = client
        .complete_messages(vec![msg], CallOptions::default())
        .await?;
    println!("{}", out.content.unwrap_or_default());

  // Optional: upload to OpenAI Files API when path provided
    if let Ok(path) = std::env::var("SUPERGLUE_UPLOAD_PATH") {
        let uploaded = client
            .upload_file(&path, FilePurpose::UserData, None)
            .await?;
        println!("uploaded file_id={}", uploaded.file_id);
    }
    Ok(())
}
