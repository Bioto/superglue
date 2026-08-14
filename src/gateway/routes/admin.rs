//! Admin REST routes (master key only).

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::Deserialize;

use crate::gateway::GatewayState;
use crate::gateway::auth::Auth;
use crate::gateway::db::UsageSummaryGroupBy;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::routes::json_ok;
use crate::providers::ProviderId;
use secrecy::ExposeSecret;

mod double_option {
    use serde::{Deserialize, Deserializer};

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        Ok(Some(Option::deserialize(deserializer)?))
    }
}

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
    #[serde(default, deserialize_with = "double_option::deserialize")]
    pub expires_at: Option<Option<String>>,
    pub metadata: Option<serde_json::Value>,
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
    #[serde(default, deserialize_with = "double_option::deserialize")]
    pub budget_id: Option<Option<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CreateBudgetBody {
    pub max_budget: f64,
    pub duration_sec: i64,
    #[serde(default = "default_enforce")]
    pub enforce: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateBudgetBody {
    pub max_budget: Option<f64>,
    pub duration_sec: Option<i64>,
    pub enforce: Option<bool>,
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
    let metadata = body.metadata.clone();
    let key = state
        .db
        .run_blocking(move |db| {
            let expires = expires.as_ref().map(|v| v.as_deref());
            let metadata = metadata.map(Some);
            db.update_api_key(&id, active, allowed_models.as_deref(), expires, metadata)
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
    let alias_owned = body.alias.clone();
    let budget_owned = body.budget_id.clone();
    let user = state
        .db
        .run_blocking(move |db| {
            let alias_update = alias_owned.as_ref().map(|a| Some(a.as_str()));
            let budget_update = budget_owned.as_ref().map(|b| b.as_deref());
            db.update_user(&id, alias_update, budget_update)
        })
        .await?;
    Ok(json_ok(user))
}

pub async fn delete_user(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let deleted_id = id.clone();
    let keys_deleted = state.db.run_blocking(move |db| db.delete_user(&id)).await?;
    Ok(json_ok(serde_json::json!({
        "deleted": deleted_id,
        "keys_deleted": keys_deleted,
    })))
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

pub async fn update_budget(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
    Json(body): Json<UpdateBudgetBody>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    if body.max_budget.is_none() && body.duration_sec.is_none() && body.enforce.is_none() {
        return Err(GatewayError::bad_request(
            "at least one of max_budget, duration_sec, or enforce is required",
        ));
    }
    let max_budget = body.max_budget;
    let duration_sec = body.duration_sec;
    let enforce = body.enforce;
    let budget = state
        .db
        .run_blocking(move |db| db.update_budget(&id, max_budget, duration_sec, enforce))
        .await?;
    Ok(json_ok(budget))
}

pub async fn delete_budget(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Path(id): Path<String>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let deleted_id = id.clone();
    let users_cleared = state
        .db
        .run_blocking(move |db| db.delete_budget(&id))
        .await?;
    Ok(json_ok(serde_json::json!({
        "deleted": deleted_id,
        "users_cleared": users_cleared,
    })))
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

#[derive(Debug, Deserialize)]
pub struct UsageSummaryQuery {
    pub user_id: Option<String>,
    pub key_id: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(default = "default_summary_group")]
    pub group_by: String,
}

fn default_summary_group() -> String {
    "day".into()
}

pub async fn usage_summary(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Query(query): Query<UsageSummaryQuery>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let group_by = UsageSummaryGroupBy::parse(&query.group_by)
        .ok_or_else(|| GatewayError::bad_request("group_by must be user, model, key, or day"))?;
    let user_id = query.user_id.clone();
    let key_id = query.key_id.clone();
    let from = query.from.clone();
    let to = query.to.clone();
    let summary = state
        .db
        .run_blocking(move |db| {
            db.usage_summary(
                user_id.as_deref(),
                key_id.as_deref(),
                from.as_deref(),
                to.as_deref(),
                group_by,
            )
        })
        .await?;
    Ok(json_ok(summary))
}

#[derive(Debug, Deserialize)]
pub struct BudgetResetsQuery {
    pub user_id: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

pub async fn list_budget_resets(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
    Query(query): Query<BudgetResetsQuery>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let user_id = query.user_id.clone();
    let limit = query.limit;
    let logs = state
        .db
        .run_blocking(move |db| db.list_budget_reset_logs(user_id.as_deref(), limit))
        .await?;
    Ok(json_ok(serde_json::json!({ "resets": logs })))
}

#[derive(Debug, serde::Serialize)]
pub struct ProviderStatus {
    pub id: String,
    pub label: String,
    pub configured: bool,
    pub base_url: String,
    pub key_suffix: Option<String>,
    pub catalog_ok: bool,
    pub model_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn list_providers(
    State(state): State<Arc<GatewayState>>,
    Auth(_auth): Auth,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let http = state.http.clone();
    let credentials = state.credentials.clone();
    let mut providers = Vec::new();
    for provider in ProviderId::ALL {
        let configured = credentials.has_key(provider);
        let base_url = credentials.base_url_for(provider);
        let key_suffix = credentials.key_for(provider).ok().map(|k| {
            let secret = k.expose_secret();
            if secret.len() <= 4 {
                secret.to_string()
            } else {
                format!("...{}", &secret[secret.len() - 4..])
            }
        });
        let (catalog_ok, model_count, error) = if configured {
            match crate::gateway::model_catalog::fetch_provider_models_for_status(
                &http,
                &credentials,
                provider,
            )
            .await
            {
                Ok(count) => (true, count, None),
                Err(e) => (false, 0, Some(e.to_string())),
            }
        } else {
            (false, 0, None)
        };
        providers.push(ProviderStatus {
            id: provider.as_str().to_string(),
            label: provider.to_string(),
            configured,
            base_url,
            key_suffix,
            catalog_ok,
            model_count,
            error,
        });
    }
    Ok(json_ok(serde_json::json!({ "providers": providers })))
}
