//! Provider-agnostic realtime voice sessions.
//!
//! This module is feature-gated because the normal SuperGlue transport remains
//! HTTP and SSE. The first provider adapter is xAI Speech to Speech.

mod error;
mod events;
mod provider;
mod session;
mod xai;

pub use error::VoiceError;
pub use events::{VoiceClientEvent, VoiceEvent};
pub use provider::{VoiceProvider, resolve_voice_provider};
pub use session::{VoiceConfig, VoiceEvents, VoiceSession};
