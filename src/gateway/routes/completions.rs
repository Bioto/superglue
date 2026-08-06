//! Chat completions and model listing routes.

use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::gateway::GatewayState;
use crate::gateway::auth::Auth;
use crate::gateway::error::GatewayError;
use crate::gateway::model_catalog;
use crate::gateway::proxy::{self, GatewayCompletionBody};

pub async fn chat_completions(
    State(state): State<Arc<GatewayState>>,
    Auth(auth): Auth,
    Json(body): Json<GatewayCompletionBody>,
) -> Result<Response, GatewayError> {
    let stream = body.completion.stream.unwrap_or(false);
    if stream {
        let byte_stream = proxy::proxy_completion_stream(
            &state.http,
            &state.credentials,
            state.db.clone(),
            auth,
            body,
        )
        .await?;
        let body = Body::from_stream(byte_stream);
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(body)
            .unwrap())
    } else {
        let (val, _request_id) =
            proxy::proxy_completion(&state.http, &state.credentials, &state.db, &auth, body)
                .await?;
        Ok(Json(val).into_response())
    }
}

pub async fn list_models(
    State(state): State<Arc<GatewayState>>,
    Auth(auth): Auth,
) -> Result<impl IntoResponse, GatewayError> {
    let data = model_catalog::list_models(&state.http, &state.credentials, &auth).await?;
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
    })))
}
