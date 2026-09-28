//! Provider-neutral events and commands for a duplex voice session.

use bytes::Bytes;
use serde_json::Value;

/// Events emitted by a connected voice provider.
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    SessionCreated {
        session_id: Option<String>,
    },
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    InputTranscript {
        text: String,
        final_: bool,
    },
    OutputAudio(Bytes),
    OutputTranscript {
        text: String,
        final_: bool,
    },
    ResponseDone,
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    Interrupted,
    Error {
        message: String,
    },
    Provider {
        event_type: String,
        payload: Value,
    },
    Closed,
}

/// Commands accepted by a voice session.
#[derive(Debug, Clone)]
pub enum VoiceClientEvent {
    Update(Value),
    Audio(Bytes),
    Text(String),
    Interrupt,
    ToolResult { call_id: String, output: Value },
    CreateResponse,
    Close,
}
