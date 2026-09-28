//! File transcription wiremock. A custom base URL uses `/v1/speech/transcriptions`.

use serde_json::json;
use superglue::audio::{TranscribeError, transcribe};
use superglue::http::{ClientConfig, HttpClient};
use superglue::providers::{ProviderCredentials, ProviderId};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_verbose_json_returns_words() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/speech/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9}
            ]
        })))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).expect("http client");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::OpenAi, "sk-test");
    creds.insert_base_url(ProviderId::OpenAi, &server.uri());

    let transcript = transcribe(
        &http,
        &creds,
        ProviderId::OpenAi,
        "clip.wav",
        b"RIFF".to_vec(),
    )
    .await
    .expect("transcribe");

    assert_eq!(transcript.text, "hello world");
    assert_eq!(transcript.words.len(), 2);
    assert_eq!(transcript.words[0].text, "hello");
}

#[tokio::test]
async fn anthropic_is_unsupported() {
    let http = HttpClient::new(ClientConfig::default()).expect("http client");
    let mut creds = ProviderCredentials::new();
    creds.insert_key(ProviderId::Anthropic, "sk-ant");

    let error = transcribe(
        &http,
        &creds,
        ProviderId::Anthropic,
        "clip.wav",
        b"RIFF".to_vec(),
    )
    .await
    .expect_err("anthropic has no file STT");

    assert!(matches!(
        error,
        TranscribeError::UnsupportedProvider(ProviderId::Anthropic)
    ));
}
