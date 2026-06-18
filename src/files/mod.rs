//! File upload helpers and per-provider Files API integration.

use std::path::Path;

use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::http::{join_base_url, Error as HttpError, HttpClient};
use crate::openai::{ChatMessage, ContentPart, FileContent, MessageContent};
use crate::providers::{
    parse_model_ref, ProviderCredentials, ProviderId, rate_limit_key_for,
};

const DEFAULT_MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;

/// OpenAI-aligned file purpose (mapped per provider where applicable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilePurpose {
    Assistants,
    UserData,
    Batch,
}

impl FilePurpose {
    #[must_use]
    pub fn as_openai_str(self) -> &'static str {
        match self {
            FilePurpose::Assistants => "assistants",
            FilePurpose::UserData => "user_data",
            FilePurpose::Batch => "batch",
        }
    }
}

/// Uploaded file metadata returned by a provider Files API.
#[derive(Debug, Clone)]
pub struct UploadedFile {
    pub provider: ProviderId,
    pub file_id: String,
    pub filename: String,
    pub bytes: usize,
    pub purpose: FilePurpose,
}

#[derive(Debug, Error)]
pub enum FileError {
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("file too large: {bytes} bytes (max {max})")]
    FileTooLarge { bytes: usize, max: usize },
    #[error("unsupported provider for file upload: {0}")]
    UnsupportedProvider(ProviderId),
    #[error(transparent)]
    Credentials(#[from] crate::providers::CredentialsError),
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Deserialize)]
struct OpenAiFileResponse {
    id: String,
    #[serde(default)]
    bytes: Option<u32>,
}

/// Upload a file to provider storage (OpenAI / xAI). Groq is unsupported; Anthropic uses inline bytes.
pub async fn upload_file(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    provider: ProviderId,
    path: &Path,
    purpose: FilePurpose,
    max_bytes: usize,
) -> Result<UploadedFile, FileError> {
    if provider == ProviderId::Groq {
        return Err(FileError::UnsupportedProvider(provider));
    }

    let bytes = std::fs::read(path)?;
    if bytes.len() > max_bytes {
        return Err(FileError::FileTooLarge {
            bytes: bytes.len(),
            max: max_bytes,
        });
    }
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();

    if provider == ProviderId::Anthropic {
        return Ok(UploadedFile {
            provider,
            file_id: String::new(),
            filename,
            bytes: bytes.len(),
            purpose,
        });
    }

    let model_ref = parse_model_ref(&format!("{}:upload", provider.as_str()));
    credentials.key_for(provider)?;
    let base_url = credentials.base_url_for(provider);
    let url = join_base_url(&base_url, "/v1/files");
    let key = credentials.key_for(provider)?;
    let rate_limit_key = rate_limit_key_for(&model_ref, credentials)?;

    let auth = format!("Bearer {}", key.expose_secret());
    let val: Value = http
        .post_multipart(
            &url,
            &[("Authorization", auth.as_str())],
            purpose.as_openai_str(),
            &filename,
            bytes,
            Some(rate_limit_key),
        )
        .await?;
    let resp: OpenAiFileResponse = serde_json::from_value(val)?;
    Ok(UploadedFile {
        provider,
        file_id: resp.id,
        filename,
        bytes: resp.bytes.unwrap_or(0) as usize,
        purpose,
    })
}

/// Minimal single-page PDF accepted by OpenAI inline file chat.
pub fn demo_pdf_bytes() -> &'static [u8] {
    b"%PDF-1.4\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
3 0 obj<</Type/Page/MediaBox[0 0 200 200]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n\
4 0 obj<</Length 44>>stream\n\
BT /F1 24 Tf 20 100 Td (Hi) Tj ET\n\
endstream\nendobj\n\
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n\
xref\n0 6\n\
0000000000 65535 f \n\
0000000010 00000 n \n\
0000000060 00000 n \n\
0000000114 00000 n \n\
0000000220 00000 n \n\
0000000314 00000 n \n\
trailer<</Size 6/Root 1 0 R>>\n\
startxref\n380\n%%EOF\n"
}

/// Build a user message with inline file bytes (base64 data URL) for chat.
#[must_use]
pub fn message_with_file_bytes(
    text: Option<&str>,
    filename: &str,
    bytes: &[u8],
) -> ChatMessage {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let mime = mime_from_filename(filename);
    let file_data = format!("data:{};base64,{}", mime, STANDARD.encode(bytes));
    let mut parts = Vec::new();
    if let Some(t) = text {
        parts.push(ContentPart::Text {
            text: t.to_string(),
        });
    }
    parts.push(ContentPart::File {
        file: FileContent {
            file_data: Some(file_data),
            file_id: None,
            filename: Some(filename.to_string()),
        },
    });
    ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContent::Parts(parts)),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: None,
    }
}

#[must_use]
pub fn default_max_upload_bytes() -> usize {
    DEFAULT_MAX_UPLOAD_BYTES
}

fn mime_from_filename(filename: &str) -> &'static str {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".pdf") {
        "application/pdf"
    } else if lower.ends_with(".txt") {
        "text/plain"
    } else if lower.ends_with(".json") {
        "application/json"
    } else {
        "application/octet-stream"
    }
}
