//! Fail-open billed-cost lookup via OpenRouter/Vercel `GET /v1/generation`.

use std::time::Duration;

use reqwest::StatusCode;
use secrecy::ExposeSecret;
use serde_json::Value;
use tokio::time::sleep;
use tracing::warn;

use crate::http::{HttpClient, join_base_url};
use crate::providers::{ProviderCredentials, ProviderId};

const GENERATION_LOOKUP_DELAYS_MS: [u64; 3] = [150, 400, 800];

/// Fetch `data.total_cost` for an OpenRouter or Vercel generation.
///
/// Returns `None` on missing credentials, empty id, HTTP failure, or a body
/// without a finite non-negative `total_cost`. A `404` retries a few times.
/// Failures never propagate: callers keep inline billed cost or the table.
pub async fn fetch_generation_cost_usd(
    http: &HttpClient,
    credentials: &ProviderCredentials,
    provider: ProviderId,
    generation_id: &str,
) -> Option<f64> {
    if !provider.exposes_generation_cost() || generation_id.is_empty() {
        return None;
    }
    let Ok(key) = credentials.key_for(provider) else {
        return None;
    };
    let url = join_base_url(
        &credentials.base_url_for(provider),
        &format!("/v1/generation?id={generation_id}"),
    );
    let auth = format!("Bearer {}", key.expose_secret());
    let headers = [("Authorization", auth.as_str())];

    for (attempt, delay_ms) in GENERATION_LOOKUP_DELAYS_MS.iter().enumerate() {
        match http.get_with_headers(&url, Some(&headers)).await {
            Ok(bytes) => return parse_generation_total_cost(&bytes),
            Err(err) if err.status() == Some(StatusCode::NOT_FOUND) => {
                if attempt + 1 == GENERATION_LOOKUP_DELAYS_MS.len() {
                    warn!(
                        generation_id,
                        provider = provider.as_str(),
                        "generation cost lookup returned 404"
                    );
                    return None;
                }
                sleep(Duration::from_millis(*delay_ms)).await;
            }
            Err(err) => {
                warn!(
                    generation_id,
                    provider = provider.as_str(),
                    status = ?err.status(),
                    "generation cost lookup failed"
                );
                return None;
            }
        }
    }
    None
}

pub(crate) fn parse_generation_total_cost(bytes: &[u8]) -> Option<f64> {
    let value: Value = serde_json::from_slice(bytes).ok()?;
    let cost = value
        .pointer("/data/total_cost")
        .or_else(|| value.pointer("/data/usage"))
        .or_else(|| value.get("total_cost"))?;
    super::json_cost_number(cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openrouter_generation_envelope() {
        let body = br#"{"data":{"id":"gen-1","total_cost":0.04}}"#;
        assert_eq!(parse_generation_total_cost(body), Some(0.04));
    }

    #[test]
    fn parses_zero_total_cost() {
        let body = br#"{"data":{"total_cost":0.0}}"#;
        assert_eq!(parse_generation_total_cost(body), Some(0.0));
    }

    #[test]
    fn rejects_negative_total_cost() {
        let body = br#"{"data":{"total_cost":-0.01}}"#;
        assert_eq!(parse_generation_total_cost(body), None);
    }
}
