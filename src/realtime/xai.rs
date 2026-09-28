//! xAI Speech to Speech WebSocket adapter.

use secrecy::ExposeSecret;
use serde_json::{Value, json};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use url::Url;

use super::error::VoiceError;
use super::provider::{VoiceProvider, VoiceWebSocket};
use super::session::VoiceConfig;

pub(crate) struct XaiVoiceProvider;

#[async_trait::async_trait]
impl VoiceProvider for XaiVoiceProvider {
    fn provider_id(&self) -> crate::providers::ProviderId {
        crate::providers::ProviderId::Xai
    }

    async fn connect(&self, config: &VoiceConfig) -> Result<VoiceWebSocket, VoiceError> {
        let key = config
            .credentials
            .key_for(crate::providers::ProviderId::Xai)?;
        let base_url = config.base_url.clone().unwrap_or_else(|| {
            config
                .credentials
                .base_url_for(crate::providers::ProviderId::Xai)
        });
        let base_url = base_url.trim_end_matches('/');
        let websocket_base_url = if let Some(value) = base_url.strip_prefix("https://") {
            format!("wss://{value}")
        } else if let Some(value) = base_url.strip_prefix("http://") {
            format!("ws://{value}")
        } else if base_url.starts_with("ws://") || base_url.starts_with("wss://") {
            base_url.to_owned()
        } else {
            return Err(VoiceError::Protocol(format!(
                "realtime base URL must use http(s) or ws(s), got {base_url}"
            )));
        };
        let mut url = Url::parse(&format!("{websocket_base_url}/v1/realtime"))
            .map_err(|error| VoiceError::Protocol(format!("invalid realtime URL: {error}")))?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("model", &config.model);
            if let Some(conversation_id) = &config.conversation_id {
                query.append_pair("conversation_id", conversation_id);
            }
        }
        let mut request = url
            .as_str()
            .into_client_request()
            .map_err(|error| VoiceError::Protocol(format!("invalid WebSocket request: {error}")))?;
        request.headers_mut().insert(
            "Authorization",
            format!("Bearer {}", key.expose_secret())
                .parse()
                .map_err(|error| {
                    VoiceError::Protocol(format!("invalid authorization header: {error}"))
                })?,
        );
        request.headers_mut().insert(
            "User-Agent",
            format!("superglue/{}", env!("CARGO_PKG_VERSION"))
                .parse()
                .map_err(|error| {
                    VoiceError::Protocol(format!("invalid user agent header: {error}"))
                })?,
        );
        // Harn also uses reqwest with aws-lc-rs. Explicitly selecting ring here
        // prevents rustls from panicking when both providers are linked.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let (socket, _) = connect_async(request).await.map_err(|error| {
            VoiceError::Protocol(format!(
                "realtime WebSocket connection failed: {error} ({url})"
            ))
        })?;
        Ok(socket)
    }
}

pub(crate) fn session_update(config: &VoiceConfig) -> Value {
    let mut session = json!({
        "voice": config.voice,
        "instructions": config.instructions,
        "turn_detection": {
            "type": if config.server_vad { "server_vad" } else { "none" }
        },
        "audio": {
            "input": {
                "format": {
                    "type": config.input_format,
                    "rate": config.sample_rate
                },
                "transport": "binary"
            },
            "output": {
                "format": {
                    "type": config.output_format,
                    "rate": config.sample_rate
                },
                "transport": "binary"
            }
        },
        "resumption": { "enabled": config.resumption_enabled }
    });
    if !config.tools.is_empty() {
        session["tools"] = Value::Array(
            config
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema
                    })
                })
                .collect(),
        );
    }
    json!({ "type": "session.update", "session": session })
}
