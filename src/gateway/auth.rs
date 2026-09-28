//! Authentication helpers for master and virtual API keys.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::gateway::db::{ApiKeyRecord, Database};
use crate::gateway::error::{GatewayError, GatewayResult};

/// Cached virtual-key auth. Keyed by SHA-256 of the raw key.
#[derive(Clone, Default)]
pub(crate) struct AuthCache {
    inner: Arc<Mutex<HashMap<[u8; 32], CachedAuth>>>,
}

#[derive(Clone)]
struct CachedAuth {
    context: AuthContext,
    expires_at: Option<String>,
}

impl AuthCache {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> GatewayResult<MutexGuard<'_, HashMap<[u8; 32], CachedAuth>>> {
        self.inner
            .lock()
            .map_err(|_| GatewayError::Internal("auth cache lock poisoned".into()))
    }

    /// Return a cached context, `None` on miss. Rejects an expired cached key.
    pub(crate) fn get(&self, hash: &[u8; 32]) -> GatewayResult<Option<AuthContext>> {
        let mut map = self.lock()?;
        let Some(entry) = map.get(hash) else {
            return Ok(None);
        };
        if let Some(expires_at) = entry.expires_at.as_deref()
            && expiry_is_past(expires_at)
        {
            map.remove(hash);
            return Err(GatewayError::unauthorized("API key has expired"));
        }
        Ok(Some(entry.context.clone()))
    }

    pub(crate) fn insert(
        &self,
        hash: [u8; 32],
        context: AuthContext,
        expires_at: Option<String>,
    ) -> GatewayResult<()> {
        self.lock()?.insert(
            hash,
            CachedAuth {
                context,
                expires_at,
            },
        );
        Ok(())
    }

    /// Drop every cached key. Call after admin key or user mutations.
    pub(crate) fn invalidate_all(&self) -> GatewayResult<()> {
        self.lock()?.clear();
        Ok(())
    }
}

/// Resolved authentication context attached to a request.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Virtual key id; `None` for master key requests.
    pub key_id: Option<String>,
    pub user_id: String,
    pub is_master: bool,
    /// `None` means unrestricted (master key only).
    pub allowed_models: Option<Vec<String>>,
    /// Max reasoning effort from key metadata; `None` means no cap.
    pub max_reasoning_effort: Option<String>,
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

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
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
    let superglue = headers.get("X-Superglue-Key").and_then(|v| v.to_str().ok());
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
    authenticate_hashed(db, &candidate, master_key_hash)
}

/// Authenticate a pre-hashed key. The caller must perform the master-key fast path.
pub fn authenticate_hashed(
    db: &Database,
    candidate: &[u8; 32],
    master_key_hash: &[u8; 32],
) -> GatewayResult<AuthContext> {
    let (context, _) = authenticate_hashed_with_expiry(db, candidate, master_key_hash)?;
    Ok(context)
}

/// Authenticate a pre-hashed key and return the key expiry for the auth cache.
pub(crate) fn authenticate_hashed_with_expiry(
    db: &Database,
    candidate: &[u8; 32],
    master_key_hash: &[u8; 32],
) -> GatewayResult<(AuthContext, Option<String>)> {
    if hashes_equal(candidate, master_key_hash) {
        return Ok((master_auth_context(), None));
    }

    let (record, models) = db
        .lookup_api_key_with_models(candidate)?
        .ok_or_else(|| GatewayError::unauthorized("invalid API key"))?;

    validate_virtual_key(&record)?;

    let expires_at = record.expires_at.clone();
    let max_reasoning_effort = record
        .metadata_json
        .as_deref()
        .and_then(parse_max_reasoning_effort);
    Ok((
        AuthContext {
            key_id: Some(record.id),
            user_id: record.user_id,
            is_master: false,
            allowed_models: Some(models),
            max_reasoning_effort,
        },
        expires_at,
    ))
}

/// Build the context for a request authenticated by the master key.
#[must_use]
pub fn master_auth_context() -> AuthContext {
    AuthContext {
        key_id: None,
        user_id: String::new(),
        is_master: true,
        allowed_models: None,
        max_reasoning_effort: None,
    }
}

fn parse_max_reasoning_effort(metadata_json: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
    value
        .get("max_reasoning_effort")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn expiry_is_past(expires_at: &str) -> bool {
    expires_at <= chrono::Utc::now().to_rfc3339().as_str()
}

fn validate_virtual_key(record: &ApiKeyRecord) -> GatewayResult<()> {
    if record.active == 0 {
        return Err(GatewayError::unauthorized("API key is inactive"));
    }
    if let Some(expires_at) = &record.expires_at
        && expiry_is_past(expires_at)
    {
        return Err(GatewayError::unauthorized("API key has expired"));
    }
    Ok(())
}

/// Resolve the effective user id for a proxied completion request.
pub fn resolve_user_id(auth: &AuthContext, body_user: Option<&str>) -> GatewayResult<String> {
    if auth.is_master {
        let user = body_user.filter(|u| !u.is_empty()).ok_or_else(|| {
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

    fn sample_context() -> AuthContext {
        AuthContext {
            key_id: Some("key-1".into()),
            user_id: "user-1".into(),
            is_master: false,
            allowed_models: Some(vec!["openai:*".into()]),
            max_reasoning_effort: None,
        }
    }

    #[test]
    fn parse_bearer_token() {
        assert_eq!(parse_bearer("Bearer abc"), Some("abc"));
        assert_eq!(parse_bearer("bearer xyz"), Some("xyz"));
        assert_eq!(parse_bearer("raw"), None);
    }

    #[test]
    fn auth_cache_hit_returns_context() {
        let cache = AuthCache::new();
        let hash = hash_key("sgw-test");
        cache.insert(hash, sample_context(), None).unwrap();
        let got = cache.get(&hash).unwrap().expect("cache hit");
        assert_eq!(got.user_id, "user-1");
        assert_eq!(got.key_id.as_deref(), Some("key-1"));
    }

    #[test]
    fn auth_cache_rejects_expired_entry() {
        let cache = AuthCache::new();
        let hash = hash_key("sgw-expired");
        cache
            .insert(
                hash,
                sample_context(),
                Some("2000-01-01T00:00:00+00:00".into()),
            )
            .unwrap();
        let err = cache.get(&hash).expect_err("expired key");
        assert!(err.to_string().contains("expired"));
        assert!(cache.get(&hash).unwrap().is_none());
    }

    #[test]
    fn auth_cache_invalidate_all_forces_miss() {
        let cache = AuthCache::new();
        let hash = hash_key("sgw-test");
        cache.insert(hash, sample_context(), None).unwrap();
        cache.invalidate_all().unwrap();
        assert!(cache.get(&hash).unwrap().is_none());
    }
}
