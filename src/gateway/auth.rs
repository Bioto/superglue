//! Authentication helpers for master and virtual API keys.

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::gateway::db::{ApiKeyRecord, Database};
use crate::gateway::error::{GatewayError, GatewayResult};

/// Resolved authentication context attached to a request.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Virtual key id; `None` for master key requests.
    pub key_id: Option<String>,
    pub user_id: String,
    pub is_master: bool,
    /// `None` means unrestricted (master key only).
    pub allowed_models: Option<Vec<String>>,
}

/// Axum extractor for [`AuthContext`] populated by route middleware.
pub struct Auth(pub AuthContext);

impl std::ops::Deref for Auth {
    type Target = AuthContext;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Send + Sync> FromRequestParts<S> for Auth {
    type Rejection = GatewayError;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<AuthContext>()
            .cloned()
            .map(Auth)
            .ok_or_else(|| GatewayError::Internal("missing auth context".into()))
    }
}

/// Hash a raw API key with SHA-256.
#[must_use]
pub fn hash_key(raw: &str) -> [u8; 32] {
    let digest = Sha256::digest(raw.as_bytes());
    digest.into()
}

/// Constant-time comparison of two 32-byte hashes.
#[must_use]
pub fn hashes_equal(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.ct_eq(b).into()
}

/// Extract bearer token from `Authorization` or `X-Superglue-Key` header value.
#[must_use]
pub fn parse_bearer(header_value: &str) -> Option<&str> {
    let trimmed = header_value.trim();
    trimmed
        .strip_prefix("Bearer ")
        .or_else(|| trimmed.strip_prefix("bearer "))
        .map(str::trim)
}

/// Extract raw key from request headers (`X-Superglue-Key` preferred).
pub fn extract_raw_key(headers: &axum::http::HeaderMap) -> GatewayResult<String> {
    let superglue = headers
        .get("X-Superglue-Key")
        .and_then(|v| v.to_str().ok());
    let authorization = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    superglue
        .or(authorization)
        .and_then(parse_bearer)
        .map(str::to_string)
        .ok_or_else(|| GatewayError::unauthorized("missing API key"))
}

/// Authenticate a raw key against the master key hash or virtual keys in the database.
pub fn authenticate(
    db: &Database,
    raw_key: &str,
    master_key_hash: &[u8; 32],
) -> GatewayResult<AuthContext> {
    let candidate = hash_key(raw_key);
    if hashes_equal(&candidate, master_key_hash) {
        return Ok(AuthContext {
            key_id: None,
            user_id: String::new(),
            is_master: true,
            allowed_models: None,
        });
    }

    let record = db
        .lookup_api_key_by_hash(&candidate)?
        .ok_or_else(|| GatewayError::unauthorized("invalid API key"))?;

    validate_virtual_key(&record)?;

    let models = db.list_key_models(&record.id)?;
    Ok(AuthContext {
        key_id: Some(record.id),
        user_id: record.user_id,
        is_master: false,
        allowed_models: Some(models),
    })
}

fn validate_virtual_key(record: &ApiKeyRecord) -> GatewayResult<()> {
    if record.active == 0 {
        return Err(GatewayError::unauthorized("API key is inactive"));
    }
    if let Some(expires_at) = &record.expires_at {
        let now = chrono::Utc::now().to_rfc3339();
        if expires_at <= &now {
            return Err(GatewayError::unauthorized("API key has expired"));
        }
    }
    Ok(())
}

/// Resolve the effective user id for a proxied completion request.
pub fn resolve_user_id(auth: &AuthContext, body_user: Option<&str>) -> GatewayResult<String> {
    if auth.is_master {
        let user = body_user
            .filter(|u| !u.is_empty())
            .ok_or_else(|| {
                GatewayError::bad_request("master key requests must include a \"user\" field")
            })?;
        return Ok(user.to_string());
    }
    Ok(auth.user_id.clone())
}

/// Require master key authentication.
pub fn require_master(auth: &AuthContext) -> GatewayResult<()> {
    if auth.is_master {
        Ok(())
    } else {
        Err(GatewayError::forbidden(
            "admin endpoints require the master key",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bearer_token() {
        assert_eq!(parse_bearer("Bearer abc"), Some("abc"));
        assert_eq!(parse_bearer("bearer xyz"), Some("xyz"));
        assert_eq!(parse_bearer("raw"), None);
    }
}
