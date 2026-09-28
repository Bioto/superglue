//! HTTP route handlers.

pub mod admin;
pub mod audio;
pub mod completions;
pub mod health;
pub mod responses;
pub mod systemone;

use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::header;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use rust_embed::Embed;

use crate::gateway::GatewayState;
use crate::gateway::auth::{
    AuthContext, authenticate_hashed_with_expiry, extract_raw_key, hash_key, hashes_equal,
    master_auth_context,
};
use crate::gateway::error::GatewayError;

/// Build the full gateway router.
pub fn router(state: Arc<GatewayState>) -> Router {
    let public = Router::new()
        .route("/health", get(health::liveness))
        .route("/health/ready", get(health::readiness))
        .route("/admin", get(admin_index))
        .route("/admin/", get(admin_index))
        .route("/admin/{*path}", get(admin_asset))
        .with_state(state.clone());

    let proxy = Router::new()
        .route("/v1/chat/completions", post(completions::chat_completions))
        .route("/v1/responses", post(responses::create_response))
        .route("/v1/models", get(completions::list_models))
        .route("/v1/audio/transcriptions", post(audio::transcribe))
        .route("/v1/speech/transcriptions", post(audio::transcribe))
        .route("/v1/systemone", post(systemone::evaluate))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
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
        .route("/v1/usage/zero-cost", delete(admin::delete_zero_cost_usage))
        .route("/v1/budget-resets", get(admin::list_budget_resets))
        .route("/v1/providers", get(admin::list_providers));
    #[cfg(feature = "capture")]
    let admin = admin
        .route(
            "/v1/capture/status",
            get(crate::gateway::capture::capture_status),
        )
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

#[derive(Embed)]
#[folder = "gateway-admin/dist/"]
struct AdminAssets;

async fn admin_index() -> Response {
    admin_asset_response("index.html")
}

async fn admin_asset(Path(path): Path<String>) -> Response {
    admin_asset_response(&path)
}

fn admin_asset_response(requested: &str) -> Response {
    let asset = AdminAssets::get(requested).or_else(|| {
        (!requested.contains('.'))
            .then(|| AdminAssets::get("index.html"))
            .flatten()
    });
    let Some(asset) = asset else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            "Gateway admin UI is not built. Run `cd gateway-admin && npm ci && npm run build`.",
        )
            .into_response();
    };
    let content_type = match requested.rsplit('.').next().unwrap_or_default() {
        "css" => "text/css; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "woff" | "woff2" => "font/woff2",
        _ => "text/html; charset=utf-8",
    };
    ([(header::CONTENT_TYPE, content_type)], asset.data).into_response()
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
    let candidate = hash_key(&raw);
    let master_key_hash = state.master_key_hash;
    if hashes_equal(&candidate, &master_key_hash) {
        return Ok(master_auth_context());
    }
    if let Some(cached) = state.auth_cache.get(&candidate)? {
        return Ok(cached);
    }
    let (context, expires_at) = state
        .db
        .run_blocking(move |db| authenticate_hashed_with_expiry(db, &candidate, &master_key_hash))
        .await?;
    state
        .auth_cache
        .insert(candidate, context.clone(), expires_at)?;
    Ok(context)
}

pub(crate) fn json_ok<T: serde::Serialize>(value: T) -> impl axum::response::IntoResponse {
    axum::Json(value)
}
