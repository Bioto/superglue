//! Offline file transcription via OpenAI-compatible `/v1/audio/transcriptions`.

use secrecy::ExposeSecret;
use serde::Deserialize;
use thiserror::Error;

use crate::http::{Error as HttpError, HttpClient, join_base_url};
use crate::providers::{
    ProviderCredentials, ProviderId, parse_model_ref, rate_limit_key_for, wire_model_id,
};

const DEFAULT_MAX_BYTES: usize = 32 * 1024 * 1024;

/// Word-level timing from a verbose transcription response.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Full-file transcription plus optional word timestamps.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<TranscriptWord>,
}

#[derive(Debug, Error)]
pub enum TranscribeError {
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error(transparent)]
    Credentials(#[from] crate::providers::CredentialsError),
    #[error("unsupported provider for file transcription: {0}")]
    UnsupportedProvider(ProviderId),
    #[error("audio too large: {bytes} bytes (max {max})")]
    FileTooLarge { bytes: usize, max: usize },
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

/// Providers that expose OpenAI-compatible file transcription.
#[must_use]
pub fn supports_transcription(provider: ProviderId) -> bool {
    matches!(provider, ProviderId::OpenAi | ProviderId::Groq)
}

/// Pick a file-STT provider from credentials.
///
/// Uses `preferred` when it supports transcription and has a key.
/// Otherwise uses OpenAI, then Groq.
#[must_use]
pub fn transcription_provider(
    preferred: ProviderId,
    credentials: &ProviderCredentials,
) -> Option<ProviderId> {
    if supports_transcription(preferred) && credentials.has_key(preferred) {
        return Some(preferred);
    }
    [ProviderId::OpenAi, ProviderId::Groq]
        .into_iter()
        .find(|provider| credentials.has_key(*provider))
}

/// Path for file transcription.
///
/// Direct provider APIs use `/v1/audio/transcriptions`.
/// A non-default base URL (the SuperGlue gateway) uses `/v1/speech/transcriptions`.
#[must_use]
pub fn transcriptions_path(base_url: &str, provider: ProviderId) -> &'static str {
    if base_url == provider.default_base_url() {
        "/v1/audio/transcriptions"
    } else {
        "/v1/speech/transcriptions"
    }
}

/// Default STT model for a provider that supports file transcription.
#[must_use]
pub fn default_transcription_model(provider: ProviderId) -> Option<&'static str> {
    match provider {
        ProviderId::OpenAi => Some("whisper-1"),
        ProviderId::Groq => Some("whisper-large-v3"),
        _ => None,
    }
}

/// Transcribe audio bytes with word timestamps when the provider returns them.
///
/// Do not log `bytes` or the resulting transcript text.
pub async fn transcribe(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    provider: ProviderId,
    filename: &str,
    bytes: Vec<u8>,
) -> Result<Transcript, TranscribeError> {
    let model = default_transcription_model(provider)
        .ok_or(TranscribeError::UnsupportedProvider(provider))?;
    if bytes.len() > DEFAULT_MAX_BYTES {
        return Err(TranscribeError::FileTooLarge {
            bytes: bytes.len(),
            max: DEFAULT_MAX_BYTES,
        });
    }

    let model_ref = parse_model_ref(&format!("{}:{model}", provider.as_str()));
    let key = credentials.key_for(provider)?;
    let base_url = credentials.base_url_for(provider);
    let url = join_base_url(&base_url, transcriptions_path(&base_url, provider));
    let rate_limit_key = rate_limit_key_for(&model_ref, credentials)?;
    let auth = format!("Bearer {}", key.expose_secret());
    let wire_model = wire_model_id(&model_ref, &base_url, provider);
    let fields = [
        ("model", wire_model.as_str()),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
    ];

    let value = http
        .post_multipart_with_fields(
            &url,
            &[("Authorization", auth.as_str())],
            &fields,
            filename,
            bytes,
            mime_from_filename(filename),
            Some(rate_limit_key),
        )
        .await?;
    parse_verbose_transcript(value)
}

fn mime_from_filename(filename: &str) -> &'static str {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".wav") {
        "audio/wav"
    } else if lower.ends_with(".mp3") {
        "audio/mpeg"
    } else {
        "application/octet-stream"
    }
}

#[derive(Debug, Deserialize)]
struct VerboseTranscriptResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Vec<VerboseWord>,
    #[serde(default)]
    segments: Vec<VerboseSegment>,
}

#[derive(Debug, Clone, Deserialize)]
struct VerboseWord {
    #[serde(default)]
    word: String,
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
}

#[derive(Debug, Deserialize)]
struct VerboseSegment {
    #[serde(default)]
    text: String,
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    #[serde(default)]
    words: Vec<VerboseWord>,
}

fn parse_verbose_transcript(value: serde_json::Value) -> Result<Transcript, TranscribeError> {
    let parsed: VerboseTranscriptResponse = serde_json::from_value(value)?;
    let mut words: Vec<TranscriptWord> = parsed
        .words
        .into_iter()
        .filter_map(word_from_verbose)
        .collect();
    if words.is_empty() {
        words = parsed
            .segments
            .iter()
            .flat_map(|segment| {
                if segment.words.is_empty() {
                    word_from_verbose(VerboseWord {
                        word: segment.text.clone(),
                        start: segment.start,
                        end: segment.end,
                    })
                    .into_iter()
                    .collect::<Vec<_>>()
                } else {
                    segment
                        .words
                        .iter()
                        .cloned()
                        .filter_map(word_from_verbose)
                        .collect()
                }
            })
            .collect();
    }
    Ok(Transcript {
        text: parsed.text,
        words,
    })
}

fn word_from_verbose(word: VerboseWord) -> Option<TranscriptWord> {
    let text = word.word.trim();
    if text.is_empty() {
        return None;
    }
    Some(TranscriptWord {
        start: word.start,
        end: word.end,
        text: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_verbose_transcript, supports_transcription};
    use crate::providers::ProviderId;
    use serde_json::json;

    #[test]
    fn openai_and_groq_support_transcription() {
        assert!(supports_transcription(ProviderId::OpenAi));
        assert!(supports_transcription(ProviderId::Groq));
        assert!(!supports_transcription(ProviderId::Anthropic));
    }

    #[test]
    fn transcriptions_path_uses_speech_on_gateway_base_url() {
        use super::transcriptions_path;

        assert_eq!(
            transcriptions_path(ProviderId::OpenAi.default_base_url(), ProviderId::OpenAi),
            "/v1/audio/transcriptions"
        );
        assert_eq!(
            transcriptions_path("https://gateway.myharn.sh", ProviderId::OpenAi),
            "/v1/speech/transcriptions"
        );
    }

    #[test]
    fn transcription_provider_prefers_chat_provider_then_openai() {
        use super::transcription_provider;
        use crate::providers::ProviderCredentials;

        let mut creds = ProviderCredentials::new();
        creds.insert_key(ProviderId::OpenAi, "sk-openai");
        creds.insert_key(ProviderId::Groq, "gsk-groq");
        assert_eq!(
            transcription_provider(ProviderId::Groq, &creds),
            Some(ProviderId::Groq)
        );
        assert_eq!(
            transcription_provider(ProviderId::OpenRouter, &creds),
            Some(ProviderId::OpenAi)
        );

        let empty = ProviderCredentials::new();
        assert_eq!(transcription_provider(ProviderId::OpenAi, &empty), None);
    }

    #[test]
    fn parse_prefers_top_level_words() {
        let transcript = parse_verbose_transcript(json!({
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9}
            ]
        }))
        .expect("parse");
        assert_eq!(transcript.text, "hello world");
        assert_eq!(transcript.words.len(), 2);
        assert_eq!(transcript.words[0].text, "hello");
    }

    #[test]
    fn parse_falls_back_to_segment_words() {
        let transcript = parse_verbose_transcript(json!({
            "text": "hi",
            "segments": [{
                "text": "hi",
                "start": 0.0,
                "end": 0.2,
                "words": [{"word": "hi", "start": 0.0, "end": 0.2}]
            }]
        }))
        .expect("parse");
        assert_eq!(transcript.words[0].text, "hi");
    }
}
