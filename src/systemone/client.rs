//! HTTP client for TypeSafe System One.

use std::time::Instant;

use reqwest::StatusCode;
use secrecy::ExposeSecret;
use thiserror::Error;
use tracing::instrument;

use super::types::{DEFAULT_MODEL, Question, State, SystemOneRequest, SystemOneResponse};
use crate::http::{Error as HttpError, HttpClient, join_base_url};
use crate::providers::{
    CredentialsError, ProviderCredentials, ProviderId, parse_model_ref, rate_limit_key_for,
};

/// Path on both TypeSafe and the SuperGlue gateway.
pub const SYSTEM_ONE_PATH: &str = "/v1/systemone";

/// Errors from a System One call.
#[derive(Debug, Error)]
pub enum SystemOneError {
    #[error(transparent)]
    Credentials(#[from] CredentialsError),
    #[error(transparent)]
    Http(#[from] HttpError),
    #[error("system one request failed validation: {0}")]
    Validation(String),
    #[error("failed to parse system one response: {0}")]
    Parse(String),
    #[error("unsupported provider for system one: {0}")]
    UnsupportedProvider(ProviderId),
}

/// Parse a model string into a TypeSafe model ref.
///
/// Bare ids such as `jev-latest` become `typesafe:jev-latest`.
pub fn qualify_model(raw: Option<&str>) -> Result<crate::providers::ModelRef, SystemOneError> {
    let raw = raw
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_MODEL);
    let model_ref = if raw.contains(':') {
        parse_model_ref(raw)
    } else {
        parse_model_ref(&format!("typesafe:{raw}"))
    };
    if !model_ref.provider.uses_system_one() {
        return Err(SystemOneError::UnsupportedProvider(model_ref.provider));
    }
    Ok(model_ref)
}

/// Model id sent on the wire.
///
/// Direct TypeSafe calls use the bare id. A gateway base URL keeps `typesafe:`.
#[must_use]
pub fn wire_system_one_model(model: &str, base_url: &str) -> String {
    let Ok(model_ref) = qualify_model(Some(model)) else {
        return model.to_string();
    };
    if base_url == ProviderId::TypeSafe.default_base_url() {
        model_ref.model
    } else {
        format!("typesafe:{}", model_ref.model)
    }
}

/// Build a request from JSON values (language bindings).
pub fn request_from_parts(
    state: serde_json::Value,
    questions: serde_json::Value,
    model: Option<String>,
) -> Result<SystemOneRequest, SystemOneError> {
    let state: State =
        serde_json::from_value(state).map_err(|err| SystemOneError::Parse(err.to_string()))?;
    let questions: std::collections::BTreeMap<String, Question> =
        serde_json::from_value(questions).map_err(|err| SystemOneError::Parse(err.to_string()))?;
    Ok(SystemOneRequest {
        state,
        model: model.filter(|value| !value.trim().is_empty()),
        questions,
    })
}

/// Evaluate `request` against TypeSafe System One.
///
/// Do not log `state` or answer values. Those can carry untrusted payload text.
#[instrument(
    skip(http, credentials, request),
    fields(model = tracing::field::Empty, request_id = tracing::field::Empty)
)]
pub async fn system_one(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    request: SystemOneRequest,
) -> Result<SystemOneResponse, SystemOneError> {
    let start = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    let model = request
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_MODEL)
        .to_string();
    let questions = request.questions.len();
    tracing::Span::current().record("model", tracing::field::display(&model));
    tracing::Span::current().record("request_id", tracing::field::display(&request_id));
    metrics::counter!(crate::telemetry::metrics::SYSTEM_ONE_TOTAL, "model" => model.clone())
        .increment(1);
    tracing::info!(
        request_id = %request_id,
        model = %model,
        questions,
        "system_one started"
    );

    match evaluate(http, credentials, request).await {
        Ok(response) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            metrics::histogram!(
                crate::telemetry::metrics::SYSTEM_ONE_DURATION_MS,
                "model" => response.model.clone()
            )
            .record(elapsed_ms);
            tracing::info!(
                request_id = %request_id,
                model = %response.model,
                questions = response.answers.len(),
                elapsed_ms,
                "system_one finished"
            );
            Ok(response)
        }
        Err(err) => {
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let error_kind = system_one_error_kind(&err);
            metrics::counter!(
                crate::telemetry::metrics::SYSTEM_ONE_ERRORS,
                "model" => model.clone(),
                "error_kind" => error_kind
            )
            .increment(1);
            tracing::warn!(
                request_id = %request_id,
                model = %model,
                questions,
                elapsed_ms,
                error_kind,
                error = %err,
                "system_one failed"
            );
            Err(err)
        }
    }
}

async fn evaluate(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    request: SystemOneRequest,
) -> Result<SystemOneResponse, SystemOneError> {
    let model_ref = qualify_model(request.model.as_deref())?;
    credentials.key_for(model_ref.provider)?;
    let base_url = credentials.base_url_for(model_ref.provider);
    let url = join_base_url(&base_url, SYSTEM_ONE_PATH);
    let key = credentials.key_for(model_ref.provider)?;
    let rate_limit_key = rate_limit_key_for(&model_ref, credentials)?;

    let mut body = request;
    body.model = Some(wire_system_one_model(
        body.model.as_deref().unwrap_or(DEFAULT_MODEL),
        &base_url,
    ));
    let payload =
        serde_json::to_value(&body).map_err(|err| SystemOneError::Parse(err.to_string()))?;

    let auth = format!("Bearer {}", key.expose_secret());
    let value = http
        .post_json_with_headers(
            &url,
            &payload,
            &[("Authorization", auth.as_str())],
            Some(rate_limit_key),
        )
        .await
        .map_err(map_http_error)?;

    serde_json::from_value(value).map_err(|err| SystemOneError::Parse(err.to_string()))
}

fn system_one_error_kind(err: &SystemOneError) -> &'static str {
    match err {
        SystemOneError::Credentials(_) => "credentials",
        SystemOneError::Http(_) => "http",
        SystemOneError::Validation(_) => "validation",
        SystemOneError::Parse(_) => "parse",
        SystemOneError::UnsupportedProvider(_) => "unsupported_provider",
    }
}

fn map_http_error(err: HttpError) -> SystemOneError {
    if err.status() == Some(StatusCode::UNPROCESSABLE_ENTITY) {
        let message = err.provider_message().unwrap_or_else(|| err.to_string());
        return SystemOneError::Validation(message);
    }
    SystemOneError::Http(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualify_defaults_to_jev_latest() {
        let model_ref = qualify_model(None).expect("default");
        assert_eq!(model_ref.provider, ProviderId::TypeSafe);
        assert_eq!(model_ref.model, DEFAULT_MODEL);
    }

    #[test]
    fn qualify_accepts_bare_and_prefixed() {
        let bare = qualify_model(Some("jev-latest")).expect("bare");
        let prefixed = qualify_model(Some("typesafe:jev-latest")).expect("prefixed");
        assert_eq!(bare.provider, ProviderId::TypeSafe);
        assert_eq!(prefixed.model, "jev-latest");
    }

    #[test]
    fn qualify_rejects_chat_providers() {
        let err = qualify_model(Some("openai:gpt-4o")).expect_err("reject");
        assert!(matches!(
            err,
            SystemOneError::UnsupportedProvider(ProviderId::OpenAi)
        ));
    }

    #[test]
    fn error_kind_labels_each_variant() {
        assert_eq!(
            system_one_error_kind(&SystemOneError::Validation("x".into())),
            "validation"
        );
        assert_eq!(
            system_one_error_kind(&SystemOneError::Parse("x".into())),
            "parse"
        );
        assert_eq!(
            system_one_error_kind(&SystemOneError::UnsupportedProvider(ProviderId::OpenAi)),
            "unsupported_provider"
        );
    }

    #[test]
    fn wire_model_strips_prefix_for_direct_api() {
        assert_eq!(
            wire_system_one_model(
                "typesafe:jev-latest",
                ProviderId::TypeSafe.default_base_url()
            ),
            "jev-latest"
        );
        assert_eq!(
            wire_system_one_model("jev-latest", "https://gateway.example.com"),
            "typesafe:jev-latest"
        );
    }
}
