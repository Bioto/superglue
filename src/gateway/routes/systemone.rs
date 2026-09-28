//! TypeSafe System One proxy.
//!
//! Public path: `POST /v1/systemone`.
//! Do not log `state` or answers.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use crate::gateway::GatewayState;
use crate::gateway::auth::{Auth, resolve_user_id};
use crate::gateway::error::GatewayError;
use crate::gateway::proxy::{preflight_async, record_usage_async};
use crate::systemone::{
    SystemOneError, SystemOneRequest, SystemOneResponse, qualify_model, system_one,
};

#[derive(Debug, Deserialize)]
pub struct GatewaySystemOneBody {
    state: crate::systemone::State,
    #[serde(default)]
    model: Option<String>,
    questions: std::collections::BTreeMap<String, crate::systemone::Question>,
    #[serde(default)]
    user: Option<String>,
}

/// Proxy a System One evaluation to TypeSafe.
pub async fn evaluate(
    State(state): State<Arc<GatewayState>>,
    Auth(auth): Auth,
    Json(body): Json<GatewaySystemOneBody>,
) -> Result<Json<Value>, GatewayError> {
    let model_ref = qualify_model(body.model.as_deref()).map_err(system_one_to_gateway)?;
    let qualified = format!("typesafe:{}", model_ref.model);
    let user_id = resolve_user_id(&auth, body.user.as_deref())?;
    preflight_async(&state.db, &auth, &user_id, &qualified).await?;

    let request = SystemOneRequest {
        state: body.state,
        model: Some(qualified.clone()),
        questions: body.questions,
    };
    let response = system_one(&state.http, &state.credentials, request)
        .await
        .map_err(system_one_to_gateway)?;

    let usage = response.usage.to_proto();
    let request_id = Uuid::new_v4().to_string();
    record_usage_async(
        &state.db,
        auth.key_id.as_deref(),
        &user_id,
        &qualified,
        &usage,
        0.0,
        &request_id,
    )
    .await?;

    let value = serde_json::to_value(SystemOneResponse {
        model: response.model,
        answers: response.answers,
        usage: response.usage,
    })
    .map_err(|err| GatewayError::Internal(err.to_string()))?;
    Ok(Json(value))
}

fn system_one_to_gateway(err: SystemOneError) -> GatewayError {
    match err {
        SystemOneError::UnsupportedProvider(provider) => {
            GatewayError::bad_request(format!("provider {provider} does not support System One"))
        }
        SystemOneError::Validation(message) => GatewayError::bad_request(message),
        SystemOneError::Credentials(err) => GatewayError::upstream(err.to_string()),
        SystemOneError::Http(err) => GatewayError::upstream(err.to_string()),
        SystemOneError::Parse(message) => GatewayError::upstream(message),
    }
}
