//! Gateway server configuration.

use std::path::PathBuf;

use secrecy::{ExposeSecret, SecretString};

/// Runtime configuration for the LLM gateway server.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub listen_addr: String,
    pub db_path: PathBuf,
    pub master_key: SecretString,
}

impl GatewayConfig {
    #[must_use]
    pub fn new(listen_addr: impl Into<String>, db_path: PathBuf, master_key: impl Into<String>) -> Self {
        Self {
            listen_addr: listen_addr.into(),
            db_path,
            master_key: SecretString::from(master_key.into()),
        }
    }

    #[must_use]
    pub fn master_key_exposed(&self) -> &str {
        self.master_key.expose_secret()
    }
}
