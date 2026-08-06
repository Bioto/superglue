//! HTTP route handlers.

pub mod admin;
pub mod completions;
pub mod health;
pub mod responses;

use std::sync::Arc;

use axum::Router;
use axum::extract::{Request, State};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, patch, post};

use crate::gateway::GatewayState;
use crate::gateway::auth::{AuthContext, authenticate, extract_raw_key};
use crate::gateway::error::GatewayError;

/// Build the full gateway router.
pub fn router(state: Arc<GatewayState>) -> Router {
    let public = Router::new()
        .route("/health", get(health::liveness))
        .route("/health/ready", get(health::readiness))
        .with_state(state.clone());

    let proxy = Router::new()
        .route("/v1/chat/completions", post(completions::chat_completions))
        .route("/v1/responses", post(responses::create_response))
        .route("/v1/models", get(completions::list_models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state.clone());

    let admin = Router::new()
        .route("/v1/keys", post(admin::create_key).get(admin::list_keys))
        .route(
            "/v1/keys/{id}",
            patch(admin::update_key).delete(admin::delete_key),
        )
        .route("/v1/users", post(admin::create_user).get(admin::list_users))
        .route(
            "/v1/users/{id}",
            patch(admin::update_user).delete(admin::delete_user),
        )
        .route(
            "/v1/budgets",
            post(admin::create_budget).get(admin::list_budgets),
        )
        .route(
            "/v1/budgets/{id}",
            patch(admin::update_budget).delete(admin::delete_budget),
        )
        .route("/v1/usage", get(admin::list_usage))
        .route("/v1/usage/summary", get(admin::usage_summary))
        .route("/v1/budget-resets", get(admin::list_budget_resets))
        .route("/v1/providers", get(admin::list_providers));
    #[cfg(feature = "capture")]
    let admin = admin
        .route("/v1/capture/status", get(crate::gateway::capture::capture_status))
        .route(
            "/v1/capture/records",
            get(crate::gateway::capture::list_capture_records),
        )
        .route(
            "/v1/capture/records/{request_id}",
            get(crate::gateway::capture::get_capture_record),
        );
    let admin = admin
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            admin_auth_middleware,
        ))
        .with_state(state.clone());

    Router::new().merge(public).merge(proxy).merge(admin)
}

async fn auth_middleware(
    State(state): State<Arc<GatewayState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, GatewayError> {
    let ctx = authenticate_request(&state, req.headers()).await?;
    req.extensions_mut().insert(ctx);
    Ok(next.run(req).await)
}

async fn admin_auth_middleware(
    State(state): State<Arc<GatewayState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, GatewayError> {
    let ctx = authenticate_request(&state, req.headers()).await?;
    crate::gateway::auth::require_master(&ctx)?;
    req.extensions_mut().insert(ctx);
    Ok(next.run(req).await)
}

async fn authenticate_request(
    state: &GatewayState,
    headers: &axum::http::HeaderMap,
) -> Result<AuthContext, GatewayError> {
    let raw = extract_raw_key(headers)?;
    let master_key_hash = state.master_key_hash;
    state
        .db
        .run_blocking(move |db| authenticate(db, &raw, &master_key_hash))
        .await
}

pub(crate) fn json_ok<T: serde::Serialize>(value: T) -> impl axum::response::IntoResponse {
    axum::Json(value)
}
