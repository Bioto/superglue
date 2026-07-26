//! Health and readiness probes.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::gateway::GatewayState;
use crate::gateway::error::GatewayError;

pub async fn liveness() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

pub async fn readiness(
    State(state): State<Arc<GatewayState>>,
) -> Result<impl IntoResponse, GatewayError> {
    state.db.ping()?;
    Ok((StatusCode::OK, "ready"))
}
