//! Per-API-key rate limit registry (shared across clients in-process).

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, OnceLock};

use crate::http::{DirectRateLimiter, direct_per_second};

use super::credentials::ApiKeyId;
use super::provider_id::ProviderId;

/// Published TypeSafe request cap is 1200/min. Cap the client at 20 QPS.
pub const TYPESAFE_DEFAULT_QPS: u32 = 20;

/// Default for providers that have no explicit cap. High enough to stay unused.
const SHARED_DEFAULT_QPS: u32 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RateLimitKey {
    pub provider: ProviderId,
    pub key_id: ApiKeyId,
}

#[derive(Debug)]
struct RegistryInner {
    limiters: HashMap<RateLimitKey, Arc<DirectRateLimiter>>,
    default_qps: NonZeroU32,
    default_qps_by_provider: HashMap<ProviderId, NonZeroU32>,
    qps_by_key_id: HashMap<ApiKeyId, NonZeroU32>,
}

/// Process-wide per-API-key QPS buckets.
#[derive(Debug, Clone)]
pub struct RateLimitRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

impl RateLimitRegistry {
    #[must_use]
    pub fn new(default_qps: NonZeroU32) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                limiters: HashMap::new(),
                default_qps,
                default_qps_by_provider: HashMap::new(),
                qps_by_key_id: HashMap::new(),
            })),
        }
    }

    pub fn set_default_qps_for_provider(&self, provider: ProviderId, qps: NonZeroU32) {
        let mut inner = self.inner.lock().expect("rate limit registry lock");
        inner.default_qps_by_provider.insert(provider, qps);
    }

    pub fn set_qps_for_key_id(&self, key_id: ApiKeyId, qps: NonZeroU32) {
        let mut inner = self.inner.lock().expect("rate limit registry lock");
        inner.qps_by_key_id.insert(key_id, qps);
    }

    /// Apply the TypeSafe published request cap. Key-specific QPS still wins.
    pub fn apply_typesafe_default(&self) {
        self.set_default_qps_for_provider(ProviderId::TypeSafe, typesafe_qps());
    }

    /// Process-wide registry so Harn, Porque, and the gateway share one TypeSafe bucket.
    #[must_use]
    pub fn shared() -> Arc<Self> {
        static REGISTRY: OnceLock<Arc<RateLimitRegistry>> = OnceLock::new();
        REGISTRY
            .get_or_init(|| {
                let registry = Self::new(nonzero_qps(SHARED_DEFAULT_QPS));
                registry.apply_typesafe_default();
                Arc::new(registry)
            })
            .clone()
    }

    /// Resolve the QPS that [`acquire`] will use for `key`.
    #[must_use]
    pub fn qps_for(&self, key: RateLimitKey) -> NonZeroU32 {
        let inner = self.inner.lock().expect("rate limit registry lock");
        resolve_qps(&inner, &key)
    }

    pub async fn acquire(&self, key: RateLimitKey) {
        let limiter = {
            let mut inner = self.inner.lock().expect("rate limit registry lock");
            if let Some(lim) = inner.limiters.get(&key) {
                Arc::clone(lim)
            } else {
                let qps = resolve_qps(&inner, &key);
                let lim = direct_per_second(qps);
                inner.limiters.insert(key, Arc::clone(&lim));
                lim
            }
        };
        limiter.until_ready().await;
    }
}

fn typesafe_qps() -> NonZeroU32 {
    nonzero_qps(TYPESAFE_DEFAULT_QPS)
}

fn nonzero_qps(qps: u32) -> NonZeroU32 {
    NonZeroU32::new(qps).expect("qps must be nonzero")
}

fn resolve_qps(inner: &RegistryInner, key: &RateLimitKey) -> NonZeroU32 {
    inner
        .qps_by_key_id
        .get(&key.key_id)
        .or_else(|| inner.default_qps_by_provider.get(&key.provider))
        .copied()
        .unwrap_or(inner.default_qps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::api_key_id;
    use secrecy::SecretString;

    #[tokio::test]
    async fn different_keys_use_independent_buckets() {
        let registry = RateLimitRegistry::new(NonZeroU32::new(10).unwrap());
        let k1 = RateLimitKey {
            provider: ProviderId::OpenAi,
            key_id: api_key_id(&SecretString::from("key-a")),
        };
        let k2 = RateLimitKey {
            provider: ProviderId::OpenAi,
            key_id: api_key_id(&SecretString::from("key-b")),
        };
        registry.acquire(k1).await;
        registry.acquire(k2).await;
        let inner = registry.inner.lock().unwrap();
        assert_eq!(inner.limiters.len(), 2);
    }

    #[tokio::test]
    async fn same_key_shares_bucket() {
        let registry = RateLimitRegistry::new(NonZeroU32::new(10).unwrap());
        let secret = SecretString::from("same-key");
        let k = RateLimitKey {
            provider: ProviderId::OpenAi,
            key_id: api_key_id(&secret),
        };
        registry.acquire(k).await;
        registry.acquire(k).await;
        let inner = registry.inner.lock().unwrap();
        assert_eq!(inner.limiters.len(), 1);
    }

    #[test]
    fn typesafe_default_is_20_qps() {
        let registry = RateLimitRegistry::new(NonZeroU32::new(10).unwrap());
        registry.apply_typesafe_default();
        let typesafe = RateLimitKey {
            provider: ProviderId::TypeSafe,
            key_id: api_key_id(&SecretString::from("ts-key")),
        };
        let openai = RateLimitKey {
            provider: ProviderId::OpenAi,
            key_id: api_key_id(&SecretString::from("oa-key")),
        };
        assert_eq!(registry.qps_for(typesafe).get(), TYPESAFE_DEFAULT_QPS);
        assert_eq!(registry.qps_for(openai).get(), 10);
    }

    #[test]
    fn shared_registry_is_process_wide() {
        let a = RateLimitRegistry::shared();
        let b = RateLimitRegistry::shared();
        assert!(Arc::ptr_eq(&a, &b));
        let typesafe = RateLimitKey {
            provider: ProviderId::TypeSafe,
            key_id: api_key_id(&SecretString::from("shared-ts")),
        };
        assert_eq!(a.qps_for(typesafe).get(), TYPESAFE_DEFAULT_QPS);
    }
}
