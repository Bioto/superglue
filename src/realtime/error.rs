//! Errors produced by realtime voice sessions.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("realtime voice sessions require a supported provider")]
    UnsupportedProvider,
    #[error("missing credentials for realtime provider: {0}")]
    MissingCredentials(#[from] crate::providers::CredentialsError),
    #[error("realtime WebSocket connection failed: {0}")]
    Connect(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("realtime voice protocol error: {0}")]
    Protocol(String),
    #[error("realtime voice session was cancelled")]
    Cancelled,
    #[error("realtime voice command channel closed")]
    Closed,
}
