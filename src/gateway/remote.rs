//! HTTP client for remote gateway admin commands.

use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::gateway::db::{
    ApiKeyListItem, BudgetRecord, UsageRecord, UserRecord,
};
use crate::gateway::error::{GatewayError, GatewayResult};

/// Admin API client for a running gateway server.
pub struct RemoteClient {
    base_url: String,
    master_key: String,
    http: Client,
}

#[derive(Debug, Deserialize)]
struct UsersResponse {
    users: Vec<UserRecord>,
}

#[derive(Debug, Deserialize)]
struct KeysResponse {
    keys: Vec<ApiKeyListItem>,
}

#[derive(Debug, Deserialize)]
struct BudgetsResponse {
    budgets: Vec<BudgetRecord>,
}

#[derive(Debug, Deserialize)]
struct UsageResponse {
    usage: Vec<UsageRecord>,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct DeleteUserResult {
    pub deleted: String,
    pub keys_deleted: u32,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
}

impl RemoteClient {
    /// Connect to a remote gateway admin API.
    pub fn new(url: &str, master_key: &str) -> GatewayResult<Self> {
        let base_url = url.trim_end_matches('/').to_string();
        if base_url.is_empty() {
            return Err(GatewayError::Internal("gateway URL is empty".into()));
        }
        if master_key.is_empty() {
            return Err(GatewayError::Internal(
                "master key is required for remote gateway commands".into(),
            ));
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| GatewayError::Internal(format!("HTTP client: {e}")))?;
        Ok(Self {
            base_url,
            master_key: master_key.to_string(),
            http,
        })
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.master_key)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> GatewayResult<T> {
        let resp = self
            .http
            .get(format!("{}{path}", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        parse_json(resp).await
    }

    async fn post_json<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> GatewayResult<T> {
        let resp = self
            .http
            .post(format!("{}{path}", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .json(body)
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        parse_json(resp).await
    }

    async fn patch_json<T: DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> GatewayResult<T> {
        let resp = self
            .http
            .patch(format!("{}{path}", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .json(body)
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        parse_json(resp).await
    }

    async fn delete(&self, path: &str) -> GatewayResult<()> {
        let resp = self
            .http
            .delete(format!("{}{path}", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            let message = error_message(resp).await;
            Err(GatewayError::Internal(format!("HTTP {status}: {message}")))
        }
    }

    pub async fn create_user(
        &self,
        user_id: &str,
        alias: Option<&str>,
        budget_id: Option<&str>,
    ) -> GatewayResult<UserRecord> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            user_id: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            alias: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            budget_id: Option<&'a str>,
        }
        self.post_json(
            "/v1/users",
            &Body {
                user_id,
                alias,
                budget_id,
            },
        )
        .await
    }

    pub async fn list_users(&self) -> GatewayResult<Vec<UserRecord>> {
        let resp: UsersResponse = self.get_json("/v1/users").await?;
        Ok(resp.users)
    }

    pub async fn update_user(
        &self,
        user_id: &str,
        alias: Option<&str>,
        budget_id: Option<&str>,
    ) -> GatewayResult<UserRecord> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            alias: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            budget_id: Option<&'a str>,
        }
        self.patch_json(
            &format!("/v1/users/{user_id}"),
            &Body { alias, budget_id },
        )
        .await
    }

    pub async fn delete_user(&self, user_id: &str) -> GatewayResult<DeleteUserResult> {
        let resp = self
            .http
            .delete(format!("{}/v1/users/{user_id}", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        parse_json(resp).await
    }

    pub async fn create_key(
        &self,
        user_id: &str,
        models: &[String],
        name: Option<&str>,
        expires_at: Option<&str>,
    ) -> GatewayResult<serde_json::Value> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<&'a str>,
            user_id: &'a str,
            allowed_models: &'a [String],
            #[serde(skip_serializing_if = "Option::is_none")]
            expires_at: Option<&'a str>,
        }
        self.post_json(
            "/v1/keys",
            &Body {
                name,
                user_id,
                allowed_models: models,
                expires_at,
            },
        )
        .await
    }

    pub async fn list_keys(&self) -> GatewayResult<Vec<ApiKeyListItem>> {
        let resp: KeysResponse = self.get_json("/v1/keys").await?;
        Ok(resp.keys)
    }

    pub async fn update_key(
        &self,
        id: &str,
        active: Option<bool>,
        models: Option<&[String]>,
        expires_at: Option<&str>,
    ) -> GatewayResult<ApiKeyListItem> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            active: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            allowed_models: Option<&'a [String]>,
            #[serde(skip_serializing_if = "Option::is_none")]
            expires_at: Option<&'a str>,
        }
        self.patch_json(
            &format!("/v1/keys/{id}"),
            &Body {
                active,
                allowed_models: models,
                expires_at,
            },
        )
        .await
    }

    pub async fn delete_key(&self, id: &str) -> GatewayResult<()> {
        self.delete(&format!("/v1/keys/{id}")).await
    }

    pub async fn create_budget(
        &self,
        max_budget: f64,
        duration_sec: i64,
        enforce: bool,
    ) -> GatewayResult<BudgetRecord> {
        #[derive(serde::Serialize)]
        struct Body {
            max_budget: f64,
            duration_sec: i64,
            enforce: bool,
        }
        self.post_json(
            "/v1/budgets",
            &Body {
                max_budget,
                duration_sec,
                enforce,
            },
        )
        .await
    }

    pub async fn list_budgets(&self) -> GatewayResult<Vec<BudgetRecord>> {
        let resp: BudgetsResponse = self.get_json("/v1/budgets").await?;
        Ok(resp.budgets)
    }

    pub async fn list_usage(
        &self,
        user_id: Option<&str>,
        key_id: Option<&str>,
        limit: u32,
    ) -> GatewayResult<Vec<UsageRecord>> {
        let mut req = self
            .http
            .get(format!("{}/v1/usage", self.base_url))
            .header("X-Superglue-Key", self.auth_header())
            .query(&[("limit", limit.to_string())]);
        if let Some(user_id) = user_id {
            req = req.query(&[("user_id", user_id)]);
        }
        if let Some(key_id) = key_id {
            req = req.query(&[("key_id", key_id)]);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        let parsed: UsageResponse = parse_json(resp).await?;
        Ok(parsed.usage)
    }

    pub async fn list_models(&self, api_key: &str) -> GatewayResult<Vec<serde_json::Value>> {
        #[derive(Debug, Deserialize)]
        struct ModelsResponse {
            data: Vec<serde_json::Value>,
        }
        let resp = self
            .http
            .get(format!("{}/v1/models", self.base_url))
            .header("X-Superglue-Key", format!("Bearer {api_key}"))
            .send()
            .await
            .map_err(|e| GatewayError::Internal(format!("request failed: {e}")))?;
        let parsed: ModelsResponse = parse_json(resp).await?;
        Ok(parsed.data)
    }
}

async fn parse_json<T: DeserializeOwned>(resp: reqwest::Response) -> GatewayResult<T> {
    if resp.status().is_success() {
        resp.json()
            .await
            .map_err(|e| GatewayError::Internal(format!("invalid JSON response: {e}")))
    } else {
        let status = resp.status();
        let message = error_message(resp).await;
        Err(GatewayError::Internal(format!("HTTP {status}: {message}")))
    }
}

async fn error_message(resp: reqwest::Response) -> String {
    resp.json::<ErrorBody>()
        .await
        .map(|b| b.error.message)
        .unwrap_or_else(|_| "request failed".into())
}
