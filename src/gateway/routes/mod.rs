//! HTTP route handlers.

pub mod admin;
pub mod completions;
pub mod health;

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, patch, post};
use axum::Router;

use crate::gateway::auth::{authenticate, extract_raw_key, AuthContext};
use crate::gateway::error::GatewayError;
use crate::gateway::GatewayState;

/// Build the full gateway router.
pub fn router(state: Arc<GatewayState>) -> Router {
    let public = Router::new()
        .route("/health", get(health::liveness))
        .route("/health/ready", get(health::readiness))
        .with_state(state.clone());

    let proxy = Router::new()
        .route("/v1/chat/completions", post(completions::chat_completions))
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
        .route("/v1/users/{id}", patch(admin::update_user))
        .route("/v1/budgets", post(admin::create_budget).get(admin::list_budgets))
        .route("/v1/usage", get(admin::list_usage))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            admin_auth_middleware,
        ))
        .with_state(state.clone());

    Router::new()
        .merge(public)
        .merge(proxy)
        .merge(admin)
}

async fn auth_middleware(
    State(state): State<Arc<GatewayState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, GatewayError> {
    let ctx = extract_auth(&state, &req)?;
    req.extensions_mut().insert(ctx);
    Ok(next.run(req).await)
}

async fn admin_auth_middleware(
    State(state): State<Arc<GatewayState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, GatewayError> {
    let ctx = extract_auth(&state, &req)?;
    crate::gateway::auth::require_master(&ctx)?;
    req.extensions_mut().insert(ctx);
    Ok(next.run(req).await)
}

fn extract_auth(state: &GatewayState, req: &Request) -> Result<AuthContext, GatewayError> {
    let raw = extract_raw_key(req.headers())?;
    authenticate(&state.db, &raw, &state.master_key_hash)
}

pub(crate) fn json_ok<T: serde::Serialize>(value: T) -> impl axum::response::IntoResponse {
    axum::Json(value)
}
