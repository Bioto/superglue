//! Per-provider API keys and base URL overrides.

use std::collections::HashMap;

use secrecy::{ExposeSecret, SecretString};
use thiserror::Error;

use super::provider_id::ProviderId;

#[derive(Debug, Error)]
pub enum CredentialsError {
    #[error("missing API key for provider {0}")]
    MissingKey(ProviderId),
}

#[derive(Debug, Clone, Default)]
pub struct ProviderCredentials {
    keys: HashMap<ProviderId, SecretString>,
    base_urls: HashMap<ProviderId, String>,
}

impl ProviderCredentials {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_key(&mut self, provider: ProviderId, key: impl Into<String>) {
        self.keys.insert(provider, SecretString::from(key.into()));
    }

    pub fn insert_base_url(&mut self, provider: ProviderId, url: impl Into<String>) {
        self.base_urls.insert(provider, url.into());
    }

    /// Load keys from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`.
    #[must_use]
    pub fn from_env() -> Self {
        let mut creds = Self::new();
        for provider in ProviderId::ALL {
            if let Ok(key) = std::env::var(provider.env_var_for_key())
                && !key.is_empty()
            {
                creds.insert_key(provider, key);
            }
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL")
            && !url.is_empty()
        {
            creds.insert_base_url(ProviderId::OpenAi, url);
        }
        creds
    }

    /// Merge a legacy single key as OpenAI when no OpenAI key is set.
    pub fn with_legacy_openai_key(&mut self, api_key: &str, base_url: Option<&str>) {
        if !api_key.is_empty() && !self.keys.contains_key(&ProviderId::OpenAi) {
            self.insert_key(ProviderId::OpenAi, api_key);
        }
        if let Some(url) = base_url
            && !url.is_empty()
            && !self.base_urls.contains_key(&ProviderId::OpenAi)
        {
            self.insert_base_url(ProviderId::OpenAi, url);
        }
    }

    pub fn key_for(&self, provider: ProviderId) -> Result<SecretString, CredentialsError> {
        self.keys
            .get(&provider)
            .cloned()
            .ok_or(CredentialsError::MissingKey(provider))
    }

    #[must_use]
    pub fn base_url_for(&self, provider: ProviderId) -> String {
        self.base_urls
            .get(&provider)
            .cloned()
            .unwrap_or_else(|| provider.default_base_url().to_string())
    }

    #[must_use]
    pub fn has_key(&self, provider: ProviderId) -> bool {
        self.keys.contains_key(&provider)
    }
}

/// Stable bucket id derived from key material (never log the raw secret).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApiKeyId(pub u64);

#[must_use]
pub fn api_key_id(key: &SecretString) -> ApiKeyId {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.expose_secret().hash(&mut hasher);
    ApiKeyId(hasher.finish())
}
