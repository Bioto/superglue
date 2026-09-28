//! A provider-neutral handle for one duplex voice session.

use base64::Engine;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::error::VoiceError;
use super::events::{VoiceClientEvent, VoiceEvent};
use super::provider::resolve_voice_provider;

/// Configuration shared by realtime provider adapters.
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    pub provider: crate::providers::ProviderId,
    pub model: String,
    pub voice: String,
    pub instructions: String,
    pub input_format: String,
    pub output_format: String,
    pub sample_rate: u32,
    pub server_vad: bool,
    pub resumption_enabled: bool,
    pub conversation_id: Option<String>,
    pub tools: Vec<crate::tools::ToolSpec>,
    pub credentials: crate::providers::ProviderCredentials,
    pub base_url: Option<String>,
    pub cancel: CancellationToken,
    pub status_emitter: Option<std::sync::Arc<crate::events::StatusEmitter>>,
}

impl VoiceConfig {
    #[must_use]
    pub fn xai(credentials: crate::providers::ProviderCredentials) -> Self {
        Self {
            provider: crate::providers::ProviderId::Xai,
            model: "grok-voice-latest".to_string(),
            voice: "eve".to_string(),
            instructions: "You are a helpful assistant.".to_string(),
            input_format: "audio/pcm".to_string(),
            output_format: "audio/pcm".to_string(),
            sample_rate: 24_000,
            server_vad: true,
            resumption_enabled: true,
            conversation_id: None,
            tools: Vec::new(),
            credentials,
            base_url: None,
            cancel: CancellationToken::new(),
            status_emitter: None,
        }
    }
}

/// A connected session. The handle is safe to move between tasks.
#[derive(Clone)]
pub struct VoiceSession {
    commands: mpsc::Sender<VoiceClientEvent>,
}

impl std::fmt::Debug for VoiceSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceSession").finish_non_exhaustive()
    }
}

/// The receiving side of a [`VoiceSession`].
pub struct VoiceEvents {
    receiver: mpsc::Receiver<VoiceEvent>,
}

impl VoiceEvents {
    pub async fn recv(&mut self) -> Option<VoiceEvent> {
        self.receiver.recv().await
    }
}

impl VoiceSession {
    /// Connect to the configured provider and start its event pump.
    ///
    /// The returned receiver owns the provider event stream. Dropping it stops
    /// delivery, while calling [`Self::close`] closes the provider connection.
    pub async fn connect(config: VoiceConfig) -> Result<(Self, VoiceEvents), VoiceError> {
        let provider = resolve_voice_provider(config.provider)?;
        let mut socket = provider.connect(&config).await?;
        socket
            .send(tokio_tungstenite::tungstenite::Message::Text(
                super::xai::session_update(&config).to_string().into(),
            ))
            .await?;

        let (mut writer, mut reader) = socket.split();
        let (command_tx, mut command_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(256);
        let cancel = config.cancel.clone();
        let status_emitter = config.status_emitter.clone();
        let model = config.model.clone();
        let request_id = uuid::Uuid::new_v4().to_string();

        tokio::spawn(async move {
            crate::events::emit_safe(
                status_emitter.as_ref(),
                crate::events::ProcessEvent::new(
                    crate::events::ProcessEventKind::VoiceSessionStart,
                    request_id.clone(),
                    model.clone(),
                ),
            )
            .await;
            let _ = event_tx
                .send(VoiceEvent::SessionCreated { session_id: None })
                .await;
            let mut decode = DecodeState::default();
            loop {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        let _ = writer.send(tokio_tungstenite::tungstenite::Message::Close(None)).await;
                        let _ = event_tx.send(VoiceEvent::Closed).await;
                        crate::events::emit_safe(
                            status_emitter.as_ref(),
                            crate::events::ProcessEvent::new(
                                crate::events::ProcessEventKind::VoiceSessionEnd,
                                request_id.clone(),
                                model.clone(),
                            ),
                        ).await;
                        break;
                    }
                    command = command_rx.recv() => {
                        match command {
                            Some(VoiceClientEvent::Close) | None => {
                                let _ = writer.send(tokio_tungstenite::tungstenite::Message::Close(None)).await;
                                let _ = event_tx.send(VoiceEvent::Closed).await;
                                crate::events::emit_safe(
                                    status_emitter.as_ref(),
                                    crate::events::ProcessEvent::new(
                                        crate::events::ProcessEventKind::VoiceSessionEnd,
                                        request_id.clone(),
                                        model.clone(),
                                    ),
                                ).await;
                                break;
                            }
                            Some(command) => {
                                if let Err(error) = send_client_event(&mut writer, command).await {
                                    let _ = event_tx.send(VoiceEvent::Error { message: error.to_string() }).await;
                                    break;
                                }
                            }
                        }
                    }
                    message = reader.next() => {
                        match message {
                            Some(Ok(message)) => {
                                if let Err(error) =
                                    forward_server_message(message, &event_tx, &mut decode).await
                                {
                                    let _ = event_tx.send(VoiceEvent::Error { message: error.to_string() }).await;
                                    break;
                                }
                            }
                            Some(Err(error)) => {
                                let _ = event_tx.send(VoiceEvent::Error { message: error.to_string() }).await;
                                break;
                            }
                            None => {
                                let _ = event_tx.send(VoiceEvent::Closed).await;
                                crate::events::emit_safe(
                                    status_emitter.as_ref(),
                                    crate::events::ProcessEvent::new(
                                        crate::events::ProcessEventKind::VoiceSessionEnd,
                                        request_id.clone(),
                                        model.clone(),
                                    ),
                                ).await;
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok((
            Self {
                commands: command_tx,
            },
            VoiceEvents { receiver: event_rx },
        ))
    }

    pub async fn update(&self, update: Value) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::Update(update)).await
    }

    pub async fn send_audio(&self, audio: impl Into<Bytes>) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::Audio(audio.into())).await
    }

    pub async fn send_text(&self, text: impl Into<String>) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::Text(text.into())).await
    }

    pub async fn interrupt(&self) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::Interrupt).await
    }

    pub async fn submit_tool_result(
        &self,
        call_id: impl Into<String>,
        output: Value,
    ) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::ToolResult {
            call_id: call_id.into(),
            output,
        })
        .await
    }

    pub async fn create_response(&self) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::CreateResponse).await
    }

    pub async fn close(&self) -> Result<(), VoiceError> {
        self.send(VoiceClientEvent::Close).await
    }

    async fn send(&self, event: VoiceClientEvent) -> Result<(), VoiceError> {
        self.commands
            .send(event)
            .await
            .map_err(|_| VoiceError::Closed)
    }
}

async fn send_client_event<W>(writer: &mut W, event: VoiceClientEvent) -> Result<(), VoiceError>
where
    W: futures_util::Sink<tokio_tungstenite::tungstenite::Message> + Unpin,
    W::Error: Into<tokio_tungstenite::tungstenite::Error>,
{
    use tokio_tungstenite::tungstenite::Message;
    let requests_response = matches!(&event, VoiceClientEvent::Text(_));
    let message = match event {
        VoiceClientEvent::Update(value) => Message::Text(value.to_string().into()),
        VoiceClientEvent::Audio(audio) => Message::Binary(audio),
        VoiceClientEvent::Text(text) => Message::Text(
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}]
                }
            })
            .to_string()
            .into(),
        ),
        VoiceClientEvent::Interrupt => {
            Message::Text(json!({"type": "response.cancel"}).to_string().into())
        }
        VoiceClientEvent::CreateResponse => {
            Message::Text(json!({"type": "response.create"}).to_string().into())
        }
        VoiceClientEvent::ToolResult { call_id, output } => Message::Text(
            json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output.to_string()
                }
            })
            .to_string()
            .into(),
        ),
        VoiceClientEvent::Close => Message::Close(None),
    };
    writer.send(message).await.map_err(Into::into)?;
    if requests_response {
        writer
            .send(Message::Text(
                json!({"type": "response.create"}).to_string().into(),
            ))
            .await
            .map_err(Into::into)?;
    }
    Ok(())
}

#[derive(Default)]
struct DecodeState {
    saw_binary_audio: bool,
}

async fn forward_server_message(
    message: tokio_tungstenite::tungstenite::Message,
    events: &mpsc::Sender<VoiceEvent>,
    decode: &mut DecodeState,
) -> Result<(), VoiceError> {
    use tokio_tungstenite::tungstenite::Message;
    match message {
        Message::Binary(bytes) => {
            decode.saw_binary_audio = true;
            events
                .send(VoiceEvent::OutputAudio(bytes))
                .await
                .map_err(|_| VoiceError::Closed)?;
        }
        Message::Text(text) => {
            let payload: Value = serde_json::from_str(&text)
                .map_err(|error| VoiceError::Protocol(error.to_string()))?;
            if let Some(event) = normalize_server_event(&payload, decode)? {
                events.send(event).await.map_err(|_| VoiceError::Closed)?;
            }
        }
        Message::Close(_) => {
            events
                .send(VoiceEvent::Closed)
                .await
                .map_err(|_| VoiceError::Closed)?;
        }
        Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
    }
    Ok(())
}

fn normalize_server_event(
    payload: &Value,
    decode: &DecodeState,
) -> Result<Option<VoiceEvent>, VoiceError> {
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| VoiceError::Protocol("server event has no type".to_string()))?;
    let text = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let event = match event_type {
        "conversation.created" => Some(VoiceEvent::SessionCreated {
            session_id: payload
                .pointer("/conversation/id")
                .and_then(Value::as_str)
                .map(str::to_string),
        }),
        "session.updated" => Some(VoiceEvent::SessionUpdated),
        "input_audio_buffer.speech_started" => Some(VoiceEvent::SpeechStarted),
        "input_audio_buffer.speech_stopped" => Some(VoiceEvent::SpeechStopped),
        "conversation.item.input_audio_transcription.completed" => {
            Some(VoiceEvent::InputTranscript {
                text: text("transcript"),
                final_: true,
            })
        }
        "response.output_audio_transcript.delta" => {
            let delta = text("delta");
            if delta.is_empty() {
                None
            } else {
                Some(VoiceEvent::OutputTranscript {
                    text: delta,
                    final_: false,
                })
            }
        }
        // OpenAI-compat aliases + *.done payloads repeat the canonical stream.
        "response.audio_transcript.delta"
        | "response.output_audio_transcript.done"
        | "response.audio_transcript.done" => None,
        "response.output_audio.delta" | "response.audio.delta" => {
            if decode.saw_binary_audio {
                None
            } else {
                let encoded = text("delta");
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|error| VoiceError::Protocol(error.to_string()))?;
                Some(VoiceEvent::OutputAudio(Bytes::from(decoded)))
            }
        }
        "response.function_call_arguments.done" => Some(VoiceEvent::FunctionCall {
            call_id: text("call_id"),
            name: text("name"),
            arguments: text("arguments"),
        }),
        "response.done" => Some(VoiceEvent::ResponseDone),
        "response.cancelled" => Some(VoiceEvent::Interrupted),
        "error" => Some(VoiceEvent::Error {
            message: payload
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("provider error")
                .to_string(),
        }),
        _ => Some(VoiceEvent::Provider {
            event_type: event_type.to_string(),
            payload: payload.clone(),
        }),
    };
    Ok(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use tokio::time::{Duration, timeout};
    use tokio_tungstenite::accept_async;
    use tokio_tungstenite::tungstenite::Message;

    #[tokio::test]
    async fn connects_and_forwards_binary_audio() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let _ = socket.next().await;
            socket
                .send(Message::Binary(Bytes::from_static(b"pcm")))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    json!({
                        "type": "conversation.created",
                        "conversation": {"id": "conversation-test"}
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
        });

        let mut credentials = crate::providers::ProviderCredentials::new();
        credentials.insert_key(crate::providers::ProviderId::Xai, "test-key");
        let mut config = VoiceConfig::xai(credentials);
        config.base_url = Some(format!("ws://{address}"));
        let (session, mut events) = VoiceSession::connect(config).await.unwrap();

        let mut audio = None;
        for _ in 0..3 {
            if let Some(VoiceEvent::OutputAudio(value)) =
                timeout(Duration::from_secs(1), events.recv())
                    .await
                    .unwrap()
            {
                audio = Some(value);
                break;
            }
        }
        assert_eq!(audio, Some(Bytes::from_static(b"pcm")));
        let _ = session.close().await;
        server.await.unwrap();
    }

    fn normalize(payload: Value) -> Option<VoiceEvent> {
        normalize_server_event(&payload, &DecodeState::default()).unwrap()
    }

    #[test]
    fn normalizes_function_calls_and_base64_audio() {
        let function = normalize(json!({
            "type": "response.function_call_arguments.done",
            "call_id": "call-1",
            "name": "search",
            "arguments": "{\"q\":\"rust\"}"
        }));
        assert!(matches!(
            function,
            Some(VoiceEvent::FunctionCall { call_id, name, .. })
                if call_id == "call-1" && name == "search"
        ));

        let audio = normalize(json!({
            "type": "response.output_audio.delta",
            "delta": "cGNt"
        }));
        assert!(
            matches!(audio, Some(VoiceEvent::OutputAudio(value)) if value == Bytes::from_static(b"pcm"))
        );

        let delta = normalize(json!({
            "type": "response.output_audio_transcript.delta",
            "delta": "Four."
        }));
        assert!(matches!(
            delta,
            Some(VoiceEvent::OutputTranscript { text, final_: false }) if text == "Four."
        ));
        let done = normalize(json!({
            "type": "response.output_audio_transcript.done",
            "transcript": "Four."
        }));
        assert!(done.is_none());
    }

    #[test]
    fn drops_compat_aliases_and_json_audio_after_binary() {
        assert!(
            normalize(json!({
                "type": "response.audio_transcript.delta",
                "delta": "Four."
            }))
            .is_none()
        );

        let decode = DecodeState {
            saw_binary_audio: true,
        };
        let skipped = normalize_server_event(
            &json!({
                "type": "response.output_audio.delta",
                "delta": "cGNt"
            }),
            &decode,
        )
        .unwrap();
        assert!(skipped.is_none());
    }
}
