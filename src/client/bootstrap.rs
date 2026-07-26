//! Shared HTTP + [`ChatOptions`] bootstrap for bindings (Py/JS/Kotlin).

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;

use crate::chat::ChatOptions;
use crate::fallback::ModelFallbackChain;
use crate::http::{ClientConfig, HttpClient, RetryPolicy};
use crate::providers::{ProviderCredentials, ProviderId, RateLimitRegistry};

use super::ClientBuildError;

/// HTTP client, chat options, and upload limit produced for language bindings.
pub struct BindingBootstrap {
    pub http: HttpClient,
    pub options: ChatOptions,
    pub max_upload_bytes: usize,
}

/// Configuration bag matching binding constructors.
#[derive(Debug, Clone)]
pub struct BindingBootstrapConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub max_tool_rounds: u32,
    pub max_retries: u32,
    pub retry_initial_delay_ms: u64,
    pub retry_max_delay_ms: u64,
    pub retry_multiplier: f64,
    pub requests_per_second: Option<u32>,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub max_output_retries: u32,
    pub pool_max_idle_per_host: usize,
    pub pool_idle_timeout: Option<Duration>,
    pub reasoning_effort: Option<String>,
    pub status_emitter: Option<Arc<crate::events::StatusEmitter>>,
    pub model_fallback: Option<ModelFallbackChain>,
    pub provider_credentials: Option<ProviderCredentials>,
    pub provider_qps: HashMap<ProviderId, u32>,
    pub max_upload_bytes: usize,
}

impl Default for BindingBootstrapConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://api.openai.com".to_string(),
            model: "gpt-5.4-nano-2026-03-17-mini".to_string(),
            system_prompt: None,
            max_tool_rounds: 16,
            max_retries: 3,
            retry_initial_delay_ms: 50,
            retry_max_delay_ms: 2000,
            retry_multiplier: 2.0,
            requests_per_second: None,
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
            max_output_retries: 3,
            pool_max_idle_per_host: 50,
            pool_idle_timeout: None,
            reasoning_effort: None,
            status_emitter: None,
            model_fallback: None,
            provider_credentials: None,
            provider_qps: HashMap::new(),
            max_upload_bytes: crate::files::default_max_upload_bytes(),
        }
    }
}

/// Build [`HttpClient`] and [`ChatOptions`] with multi-provider credentials and per-key rate limits.
pub fn bootstrap_from_parts(
    config: BindingBootstrapConfig,
) -> Result<BindingBootstrap, ClientBuildError> {
    let api_key = config.api_key;
    if api_key.is_empty() {
        return Err(ClientBuildError::MissingApiKey);
    }

    let mut creds = config
        .provider_credentials
        .unwrap_or_else(ProviderCredentials::from_env);
    creds.with_legacy_openai_key(&api_key, Some(&config.base_url));

    let rate_limit_registry =
        if let Some(qps) = config.requests_per_second.and_then(NonZeroU32::new) {
            let reg = RateLimitRegistry::new(qps);
            for (provider, qps) in &config.provider_qps {
                if let Some(nz) = NonZeroU32::new(*qps) {
                    reg.set_default_qps_for_provider(*provider, nz);
                }
            }
            Some(Arc::new(reg))
        } else {
            None
        };

    let http_cfg = ClientConfig {
        retry: RetryPolicy {
            max_retries: config.max_retries,
            initial_interval_ms: config.retry_initial_delay_ms,
            max_interval_ms: config.retry_max_delay_ms,
            multiplier: config.retry_multiplier,
        },
        quota_per_second: None,
        rate_limit_registry,
        timeout: config.timeout,
        connect_timeout: config.connect_timeout,
        pool_max_idle_per_host: config.pool_max_idle_per_host,
        pool_idle_timeout: config.pool_idle_timeout,
        ..ClientConfig::default()
    };
    let http = HttpClient::new(http_cfg)?;

    let options = ChatOptions {
        api_key: SecretString::from(api_key),
        model: config.model,
        base_url: config.base_url,
        system_prompt: config.system_prompt,
        max_tool_rounds: config.max_tool_rounds,
        reasoning_effort: config.reasoning_effort,
        status_emitter: config.status_emitter,
        model_fallback: config.model_fallback,
        provider_credentials: Some(Arc::new(creds)),
        ..Default::default()
    };

    Ok(BindingBootstrap {
        http,
        options,
        max_upload_bytes: config.max_upload_bytes,
    })
}

/// Parse provider name strings (`openai`, `anthropic`, …) for binding maps.
pub fn provider_id_from_str(s: &str) -> Option<ProviderId> {
    s.parse().ok()
}
