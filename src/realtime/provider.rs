//! Provider boundary for realtime sessions.

use async_trait::async_trait;
use tokio::net::TcpStream;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use super::error::VoiceError;
use super::session::VoiceConfig;

pub(crate) type VoiceWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[async_trait]
pub trait VoiceProvider: Send + Sync {
    fn provider_id(&self) -> crate::providers::ProviderId;

    async fn connect(&self, config: &VoiceConfig) -> Result<VoiceWebSocket, VoiceError>;
}

pub fn resolve_voice_provider(
    provider: crate::providers::ProviderId,
) -> Result<Box<dyn VoiceProvider>, VoiceError> {
    match provider {
        crate::providers::ProviderId::Xai => Ok(Box::new(super::xai::XaiVoiceProvider)),
        _ => Err(VoiceError::UnsupportedProvider),
    }
}
