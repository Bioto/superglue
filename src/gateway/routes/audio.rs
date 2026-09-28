//! Offline file transcription proxy.
//!
//! Public paths: `/v1/audio/transcriptions` and `/v1/speech/transcriptions`.
//! Upstream providers still receive `/v1/audio/transcriptions`.
//!
//! Do not log audio bytes or transcript text.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use axum::response::{IntoResponse, Response};
use secrecy::ExposeSecret;

use crate::audio::supports_transcription;
use crate::gateway::GatewayState;
use crate::gateway::auth::{Auth, resolve_user_id};
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::proxy::preflight_async;
use crate::http::join_base_url;
use crate::providers::parse_model_ref;

const MAX_AUDIO_BYTES: usize = 32 * 1024 * 1024;

struct TranscriptionForm {
    model: String,
    user: Option<String>,
    fields: Vec<(String, String)>,
    filename: String,
    file_bytes: Vec<u8>,
    mime: String,
}

/// Proxy an OpenAI-compatible transcription request to the upstream provider.
pub async fn transcribe(
    State(state): State<Arc<GatewayState>>,
    Auth(auth): Auth,
    multipart: Multipart,
) -> Result<Response, GatewayError> {
    let form = parse_form(multipart).await?;
    let model_ref = parse_model_ref(&form.model);
    if !supports_transcription(model_ref.provider) {
        return Err(GatewayError::bad_request(format!(
            "file transcription is not supported for {}",
            model_ref.provider
        )));
    }
    let qualified = qualify_transcription_model(&form.model);
    let user_id = resolve_user_id(&auth, form.user.as_deref())?;
    preflight_async(&state.db, &auth, &user_id, &qualified).await?;

    let key = state
        .credentials
        .key_for(model_ref.provider)
        .map_err(|error| GatewayError::upstream(error.to_string()))?;
    let base_url = state.credentials.base_url_for(model_ref.provider);
    let url = join_base_url(&base_url, "/v1/audio/transcriptions");
    let auth_header = format!("Bearer {}", key.expose_secret());

    let mut upstream_fields = form.fields;
    upstream_fields.retain(|(name, _)| name != "model");
    upstream_fields.insert(0, ("model".to_string(), model_ref.model.clone()));
    let field_refs: Vec<(&str, &str)> = upstream_fields
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();

    let value = state
        .http
        .post_multipart_with_fields(
            &url,
            &[("Authorization", auth_header.as_str())],
            &field_refs,
            &form.filename,
            form.file_bytes,
            &form.mime,
            None,
        )
        .await
        .map_err(|error| GatewayError::upstream(error.to_string()))?;
    Ok(Json(value).into_response())
}

fn qualify_transcription_model(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.contains(':') {
        return trimmed.to_string();
    }
    let model_ref = parse_model_ref(trimmed);
    format!("{}:{}", model_ref.provider.as_str(), model_ref.model)
}

async fn parse_form(mut multipart: Multipart) -> GatewayResult<TranscriptionForm> {
    let mut model = None;
    let mut user = None;
    let mut fields = Vec::new();
    let mut file = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| GatewayError::bad_request(error.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            let filename = field
                .file_name()
                .filter(|name| !name.is_empty())
                .unwrap_or("audio.wav")
                .to_string();
            let mime = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_string();
            let bytes = field
                .bytes()
                .await
                .map_err(|error| GatewayError::bad_request(error.to_string()))?;
            if bytes.len() > MAX_AUDIO_BYTES {
                return Err(GatewayError::payload_too_large("audio body exceeds 32 MiB"));
            }
            if bytes.is_empty() {
                return Err(GatewayError::bad_request("audio file is empty"));
            }
            file = Some((filename, bytes.to_vec(), mime));
            continue;
        }

        let text = field
            .text()
            .await
            .map_err(|error| GatewayError::bad_request(error.to_string()))?;
        match name.as_str() {
            "model" => model = Some(text),
            "user" => user = Some(text).filter(|value| !value.is_empty()),
            _ if !name.is_empty() => fields.push((name, text)),
            _ => {}
        }
    }

    let model = model
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| GatewayError::bad_request("model is required"))?;
    let (filename, file_bytes, mime) =
        file.ok_or_else(|| GatewayError::bad_request("file is required"))?;
    Ok(TranscriptionForm {
        model,
        user,
        fields,
        filename,
        file_bytes,
        mime,
    })
}

#[cfg(test)]
mod tests {
    use super::qualify_transcription_model;

    #[test]
    fn qualify_adds_openai_prefix_to_bare_whisper() {
        assert_eq!(qualify_transcription_model("whisper-1"), "openai:whisper-1");
        assert_eq!(
            qualify_transcription_model("openai:whisper-1"),
            "openai:whisper-1"
        );
    }
}
