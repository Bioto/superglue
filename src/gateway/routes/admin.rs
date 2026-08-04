//! Admin REST routes (master key only).

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::Deserialize;

use crate::gateway::GatewayState;
use crate::gateway::auth::Auth;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::routes::json_ok;

#[derive(Debug, Deserialize)]
pub struct CreateKeyBody {
    pub name: Option<String>,
    pub user_id: String,
    pub allowed_models: Vec<String>,
    pub expires_at: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateKeyBody {
    pub active: Option<bool>,
    pub allowed_models: Option<Vec<String>>,
    pub expires_at: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateUserBody {
    pub user_id: String,
    pub alias: Option<String>,
    pub budget_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateUserBody {
    pub alias: Option<String>,
    pub budget_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBudgetBody {
    pub max_budget: f64,
    pub duration_sec: i64,
    #[serde(default = "default_enforce")]
    pub enforce: bool,
}

fn default_enforce() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    pub user_id: Option<String>,
    pub key_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    100
}

pub async fn create_key(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Json(body): Json<CreateKeyBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    if body.user_id.is_empty() {
        return Err(GatewayError::bad_request("user_id is required"));
    }
    let metadata_json = body
        .metadata
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| GatewayError::bad_request(e.to_string()))?;
    let name = body.name.clone();
    let user_id = body.user_id.clone();
    let allowed_models = body.allowed_models.clone();
    let expires_at = body.expires_at.clone();
    let metadata_json = metadata_json.clone();
    let result = state
        .db
        .run_blocking(move |db| {
            db.create_api_key(
                name.as_deref(),
                &user_id,
                &allowed_models,
                expires_at.as_deref(),
                metadata_json.as_deref(),
            )
        })
        .await?;
    Ok(json_ok(serde_json::json!({
        "id": result.id,
        "key": result.plaintext_key,
        "key_prefix": result.key_prefix,
        "user_id": body.user_id,
        "allowed_models": body.allowed_models,
    })))
}

pub async fn list_keys(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let keys = state.db.run_blocking(|db| db.list_api_keys()).await?;
    Ok(json_ok(serde_json::json!({ "keys": keys })))
}

pub async fn update_key(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
    Json(body): Json<UpdateKeyBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let expires = body.expires_at.clone();
    let allowed_models = body.allowed_models.clone();
    let active = body.active;
    let key = state
        .db
        .run_blocking(move |db| {
            let expires = expires.as_ref().map(|v| Some(v.as_str()));
            db.update_api_key(&id, active, allowed_models.as_deref(), expires)
        })
        .await?;
    Ok(json_ok(key))
}

pub async fn delete_key(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let deleted_id = id.clone();
    state
        .db
        .run_blocking(move |db| db.delete_api_key(&id))
        .await?;
    Ok(json_ok(serde_json::json!({ "deleted": deleted_id })))
}

pub async fn create_user(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Json(body): Json<CreateUserBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let user_id = body.user_id.clone();
    let alias = body.alias.clone();
    let budget_id = body.budget_id.clone();
    let user = state
        .db
        .run_blocking(move |db| db.create_user(&user_id, alias.as_deref(), budget_id.as_deref()))
        .await?;
    Ok(json_ok(user))
}

pub async fn list_users(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let users = state.db.run_blocking(|db| db.list_users()).await?;
    Ok(json_ok(serde_json::json!({ "users": users })))
}

pub async fn update_user(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
    Json(body): Json<UpdateUserBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let alias = body.alias.clone();
    let budget_id = body.budget_id.clone();
    let user = state
        .db
        .run_blocking(move |db| {
            db.update_user(&id, Some(alias.as_deref()), Some(budget_id.as_deref()))
        })
        .await?;
    Ok(json_ok(user))
}

pub async fn create_budget(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Json(body): Json<CreateBudgetBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let budget = state
        .db
        .run_blocking(move |db| db.create_budget(body.max_budget, body.duration_sec, body.enforce))
        .await?;
    Ok(json_ok(budget))
}

pub async fn list_budgets(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let budgets = state.db.run_blocking(|db| db.list_budgets()).await?;
    Ok(json_ok(serde_json::json!({ "budgets": budgets })))
}

pub async fn list_usage(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Query(query): Query<UsageQuery>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let user_id = query.user_id.clone();
    let key_id = query.key_id.clone();
    let limit = query.limit;
    let logs = state
        .db
        .run_blocking(move |db| db.list_usage(user_id.as_deref(), key_id.as_deref(), limit))
        .await?;
    Ok(json_ok(serde_json::json!({ "usage": logs })))
}
