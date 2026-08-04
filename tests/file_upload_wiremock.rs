//! File upload wiremock: POST /v1/files then chat with file_id.

use serde_json::json;
use superglue::files::{FilePurpose, upload_file};
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::{ChatMessage, ContentPart, FileContent, MessageContent};
use superglue::providers::ProviderCredentials;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn openai_upload_returns_file_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/files"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "file-abc",
            "bytes": 12,
        })))
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let mut creds = ProviderCredentials::new();
    creds.insert_key(superglue::providers::ProviderId::OpenAi, "sk-test");
    creds.insert_base_url(superglue::providers::ProviderId::OpenAi, &server.uri());

    let uploaded = upload_file(
        &http,
        &creds,
        superglue::providers::ProviderId::OpenAi,
        std::path::Path::new("/dev/null"),
        FilePurpose::UserData,
        1024,
    )
    .await
    .unwrap();

    assert_eq!(uploaded.file_id, "file-abc");
}

#[tokio::test]
async fn anthropic_inline_document_in_request() {
    use superglue::chat::{ChatOptions, complete_with_tools};
    use superglue::guardrails::GuardrailRegistry;
    use superglue::hooks::HookRegistry;
    use superglue::tools::ToolRegistry;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [{"type": "text", "text": "summary"}],
            "stop_reason": "end_turn",
        })))
        .mount(&server)
        .await;

    let mut creds = ProviderCredentials::new();
    creds.insert_key(superglue::providers::ProviderId::Anthropic, "sk-ant");
    creds.insert_base_url(superglue::providers::ProviderId::Anthropic, &server.uri());

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("legacy"),
        model: "anthropic:claude-sonnet-4-20250514".into(),
        max_tool_rounds: 1,
        provider_credentials: Some(std::sync::Arc::new(creds)),
        ..Default::default()
    };

    let msg = ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Parts(vec![
            ContentPart::Text {
                text: "Summarize".into(),
            },
            ContentPart::File {
                file: FileContent {
                    file_data: Some("JVBERi0=".into()),
                    file_id: None,
                    filename: Some("doc.pdf".into()),
                },
            },
        ])),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
        provider_blocks: None,
    };

    let out = complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![msg],
        &opts,
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("summary"));
}
