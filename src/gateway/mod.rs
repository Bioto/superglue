//! LLM gateway server — OpenAI-compatible proxy with keys, budgets, and usage tracking.

mod auth;
mod budget;
#[cfg(feature = "capture")]
pub mod capture;
pub mod cli;
mod config;
mod db;
mod error;
mod model_access;
mod model_catalog;
mod proxy;
pub mod remote;
mod routes;

pub use routes::router;

pub use cli::{DbArgs, GatewayCommand, OutputFormat};
pub use config::GatewayConfig;
pub use error::GatewayError;

use std::net::SocketAddr;
use std::sync::Arc;

use tracing::info;

use crate::gateway::auth::{AuthCache, hash_key};
use crate::gateway::db::Database;
use crate::http::{ClientConfig, HttpClient};
use crate::providers::ProviderCredentials;

#[cfg(feature = "capture")]
use crate::gateway::capture::{CaptureRuntime, spawn_capture_pipeline};

/// Shared state for all gateway HTTP handlers.
pub struct GatewayState {
    pub config: GatewayConfig,
    pub db: Database,
    pub http: Arc<HttpClient>,
    pub credentials: Arc<ProviderCredentials>,
    pub master_key_hash: [u8; 32],
    pub(crate) auth_cache: AuthCache,
    #[cfg(feature = "capture")]
    pub capture: Option<Arc<CaptureRuntime>>,
}

impl GatewayState {
    /// Capture sink for proxy paths, when capture is enabled.
    #[cfg(feature = "capture")]
    #[must_use]
    pub fn capture_sink(&self) -> Option<Arc<crate::gateway::capture::CaptureSink>> {
        self.capture.as_ref().map(|runtime| runtime.sink())
    }
    /// Build gateway state from configuration, opening the database and loading provider credentials.
    pub fn new(config: GatewayConfig) -> Result<Self, GatewayError> {
        Self::with_credentials(config, ProviderCredentials::from_env())
    }

    /// Build gateway state with explicit provider credentials (used in tests).
    pub fn with_credentials(
        config: GatewayConfig,
        credentials: ProviderCredentials,
    ) -> Result<Self, GatewayError> {
        let db = Database::open(&config.db_path)?;
        let http = Arc::new(
            HttpClient::new(ClientConfig::for_llm())
                .map_err(|e| GatewayError::Internal(e.to_string()))?,
        );
        let master_key_hash = hash_key(config.master_key_exposed());
        Ok(Self {
            config,
            db,
            http,
            credentials: Arc::new(credentials),
            master_key_hash,
            auth_cache: AuthCache::new(),
            #[cfg(feature = "capture")]
            capture: None,
        })
    }

    /// Start capture pipeline when configured (async init from [`serve`]).
    #[cfg(feature = "capture")]
    pub async fn init_capture(&mut self) -> Result<(), GatewayError> {
        if let Some(cfg) = self.config.capture.clone() {
            self.capture = Some(spawn_capture_pipeline(cfg).await?);
        }
        Ok(())
    }
}

/// Start the axum HTTP server on the configured listen address.
pub async fn serve(config: GatewayConfig) -> Result<(), GatewayError> {
    let mut state = GatewayState::new(config)?;
    #[cfg(feature = "capture")]
    state.init_capture().await?;
    let state = Arc::new(state);
    let addr: SocketAddr = state
        .config
        .listen_addr
        .parse()
        .map_err(|e| GatewayError::Internal(format!("invalid listen address: {e}")))?;
    let app = router(state.clone());

    info!(%addr, "superglue gateway server starting");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| GatewayError::Internal(format!("failed to bind: {e}")))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| GatewayError::Internal(format!("server error: {e}")))?;
    Ok(())
}

/// Build test gateway state with a temporary database (integration tests).
pub fn test_state(master_key: &str, db_path: &std::path::Path) -> Arc<GatewayState> {
    let config = GatewayConfig::new("127.0.0.1:0", db_path.to_path_buf(), master_key);
    Arc::new(GatewayState::new(config).expect("test gateway state"))
}

/// Test state with OpenAI credentials pointed at a mock server.
pub fn test_state_with_openai(
    master_key: &str,
    db_path: &std::path::Path,
    base_url: &str,
    api_key: &str,
) -> Arc<GatewayState> {
    test_state_with_provider(
        master_key,
        db_path,
        crate::providers::ProviderId::OpenAi,
        base_url,
        api_key,
    )
}

/// Test state with one provider pointed at a mock server.
pub fn test_state_with_provider(
    master_key: &str,
    db_path: &std::path::Path,
    provider: crate::providers::ProviderId,
    base_url: &str,
    api_key: &str,
) -> Arc<GatewayState> {
    use crate::providers::ProviderCredentials;
    let config = GatewayConfig::new("127.0.0.1:0", db_path.to_path_buf(), master_key);
    let mut credentials = ProviderCredentials::new();
    credentials.insert_key(provider, api_key);
    credentials.insert_base_url(provider, base_url);
    Arc::new(GatewayState::with_credentials(config, credentials).expect("test gateway state"))
}

#[cfg(feature = "capture")]
pub async fn test_state_with_openai_and_capture(
    master_key: &str,
    db_path: &std::path::Path,
    base_url: &str,
    api_key: &str,
    capture: crate::gateway::capture::CaptureConfig,
) -> Arc<GatewayState> {
    use crate::providers::{ProviderCredentials, ProviderId};
    let mut config = GatewayConfig::new("127.0.0.1:0", db_path.to_path_buf(), master_key);
    config.capture = Some(capture);
    let mut credentials = ProviderCredentials::new();
    credentials.insert_key(ProviderId::OpenAi, api_key);
    credentials.insert_base_url(ProviderId::OpenAi, base_url);
    let mut state =
        GatewayState::with_credentials(config, credentials).expect("test gateway state");
    state.init_capture().await.expect("init capture");
    Arc::new(state)
}
