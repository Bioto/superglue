//! Upstream model catalog for unrestricted keys and provider wildcards.

use std::collections::HashSet;
use std::str::FromStr;

use secrecy::ExposeSecret;
use serde_json::{Value, json};
use tracing::warn;

use crate::gateway::auth::AuthContext;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::model_access::is_unrestricted;
use crate::http::{HttpClient, join_base_url};
use crate::providers::{ProviderCredentials, ProviderId};

fn model_entry(id: &str) -> Value {
    json!({
        "id": id,
        "object": "model",
        "owned_by": "superglue-gateway"
    })
}

fn dedupe_sort_models(models: &mut Vec<Value>) {
    models.sort_by(|a, b| {
        a["id"]
            .as_str()
            .unwrap_or_default()
            .cmp(b["id"].as_str().unwrap_or_default())
    });
    models.dedup_by(|a, b| a["id"] == b["id"]);
}

/// List models visible to the caller, fetching upstream catalogs when unrestricted.
pub async fn list_models(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    auth: &AuthContext,
) -> GatewayResult<Vec<Value>> {
    if is_unrestricted(&auth.allowed_models) {
        return list_all_provider_models(http, credentials).await;
    }

    let patterns = auth.allowed_models.as_deref().unwrap_or(&[]);
    if patterns.iter().any(|p| p == "*") {
        return list_all_provider_models(http, credentials).await;
    }

    let mut result = Vec::new();
    let mut fetch_providers = HashSet::new();

    for pattern in patterns {
        if let Some(prefix) = pattern.strip_suffix(":*") {
            if let Ok(provider) = ProviderId::from_str(prefix) {
                fetch_providers.insert(provider);
                continue;
            }
        }
        result.push(model_entry(pattern));
    }

    for provider in fetch_providers {
        match fetch_provider_models(http, credentials, provider).await {
            Ok(models) if !models.is_empty() => result.extend(models),
            _ => result.push(model_entry(&format!("{}:*", provider.as_str()))),
        }
    }

    dedupe_sort_models(&mut result);
    Ok(result)
}

async fn list_all_provider_models(
    http: &HttpClient,
    credentials: &ProviderCredentials,
) -> GatewayResult<Vec<Value>> {
    let mut result = Vec::new();
    for provider in ProviderId::ALL {
        if !credentials.has_key(provider) {
            continue;
        }
        match fetch_provider_models(http, credentials, provider).await {
            Ok(mut models) => result.append(&mut models),
            Err(e) => warn!(provider = %provider, error = %e, "failed to list upstream models"),
        }
    }
    dedupe_sort_models(&mut result);
    Ok(result)
}

async fn fetch_provider_models(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    provider: ProviderId,
) -> GatewayResult<Vec<Value>> {
    let key = credentials
        .key_for(provider)
        .map_err(|e| GatewayError::Internal(e.to_string()))?;
    let base = credentials.base_url_for(provider);
    let url = join_base_url(&base, "/v1/models");
    let bearer = format!("Bearer {}", key.expose_secret());

    let headers: Vec<(&str, &str)> = match provider {
        ProviderId::Anthropic => vec![
            ("x-api-key", key.expose_secret()),
            ("anthropic-version", "2023-06-01"),
        ],
        _ => vec![("Authorization", bearer.as_str())],
    };

    let bytes = http
        .get_with_headers(&url, Some(&headers))
        .await
        .map_err(|e| GatewayError::upstream(e.to_string()))?;
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|e| GatewayError::Internal(format!("invalid models JSON: {e}")))?;

    let models = body["data"]
        .as_array()
        .ok_or_else(|| GatewayError::upstream("upstream models response missing data array"))?;

    Ok(models
        .iter()
        .filter_map(|entry| {
            let id = entry["id"].as_str()?;
            Some(model_entry(&format!("{}:{id}", provider.as_str())))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::auth::AuthContext;

    fn unrestricted_auth() -> AuthContext {
        AuthContext {
            key_id: None,
            user_id: String::new(),
            is_master: true,
            allowed_models: None,
        }
    }

    #[test]
    fn literal_patterns_are_returned_without_fetch() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let auth = AuthContext {
            key_id: Some("k1".into()),
            user_id: "u1".into(),
            is_master: false,
            allowed_models: Some(vec![
                "openai:gpt-4o-mini".into(),
                "anthropic:claude-3-5-sonnet-20241022".into(),
            ]),
        };
        let http = HttpClient::new(crate::http::ClientConfig::default()).unwrap();
        let creds = ProviderCredentials::new();
        let models = rt
            .block_on(list_models(&http, &creds, &auth))
            .unwrap();
        let ids: Vec<_> = models
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        assert!(ids.contains(&"openai:gpt-4o-mini"));
        assert!(ids.contains(&"anthropic:claude-3-5-sonnet-20241022"));
        assert!(!ids.contains(&"*"));
    }

    #[tokio::test]
    async fn unrestricted_without_credentials_is_empty() {
        let http = HttpClient::new(crate::http::ClientConfig::default()).unwrap();
        let creds = ProviderCredentials::new();
        let models = list_models(&http, &creds, &unrestricted_auth())
            .await
            .unwrap();
        assert!(models.is_empty());
    }
}
