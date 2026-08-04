//! OpenAI Responses API proxy route.

use std::sync::Arc;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::gateway::GatewayState;
use crate::gateway::auth::Auth;
use crate::gateway::error::GatewayError;
use crate::gateway::proxy::{self, GatewayResponsesBody};

pub async fn create_response(
    State(state): State<Arc<GatewayState>>,
    Auth(auth): Auth,
    Json(body): Json<GatewayResponsesBody>,
) -> Result<Response, GatewayError> {
    let stream = body.stream.unwrap_or(false);
    if stream {
        let byte_stream = proxy::proxy_response_stream(
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
            proxy::proxy_response(&state.http, &state.credentials, &state.db, &auth, body).await?;
        Ok(Json(val).into_response())
    }
}
