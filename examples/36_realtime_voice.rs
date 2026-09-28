//! Minimal realtime voice client.
//!
//! Run with:
//! `XAI_API_KEY=... cargo run --features realtime --example 36_realtime_voice`

use std::sync::Arc;

use bytes::Bytes;
use superglue::realtime::{VoiceConfig, VoiceEvent, VoiceSession};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let credentials = superglue::providers::ProviderCredentials::from_env();
    let _ = credentials.key_for(superglue::providers::ProviderId::Xai)?;
    let config = VoiceConfig::xai(credentials);
    let cancel = config.cancel.clone();
    let (session, mut events) = VoiceSession::connect(config).await?;

    let reader = BufReader::new(tokio::io::stdin());
    let mut lines = reader.lines();
    let sender = Arc::new(session);
    let input_sender = Arc::clone(&sender);
    tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            if line == "/quit" {
                cancel.cancel();
                break;
            }
            if let Err(error) = input_sender.send_text(line).await {
                eprintln!("send error: {error}");
                break;
            }
        }
    });

    while let Some(event) = events.recv().await {
        match event {
            VoiceEvent::OutputAudio(audio) => write_audio(audio).await?,
            VoiceEvent::InputTranscript { text, final_ } => {
                println!("you{}: {text}", if final_ { "" } else { "..." });
            }
            VoiceEvent::OutputTranscript { text, final_ } => {
                println!("assistant{}: {text}", if final_ { "" } else { "..." });
            }
            VoiceEvent::FunctionCall {
                call_id,
                name,
                arguments,
            } => println!("tool call {name} ({call_id}): {arguments}"),
            VoiceEvent::Error { message } => eprintln!("voice error: {message}"),
            VoiceEvent::Closed => break,
            _ => {}
        }
    }
    Ok(())
}

async fn write_audio(audio: Bytes) -> Result<(), std::io::Error> {
    // The example keeps the transport visible without choosing a platform
    // audio backend. Pipe stdout to a PCM player when testing.
    tokio::io::stdout().write_all(&audio).await
}
