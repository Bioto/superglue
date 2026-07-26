//! Shared helpers for numbered `cargo run --example` binaries.

pub mod context_tools;

use std::process;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use superglue::chat::CompletionOutcome;
use superglue::tools::{Tool, ToolInvokeError, ToolSpec};
use superglue::{Client, ClientBuildError};

pub const DEFAULT_MODEL: &str = "gpt-5.4-nano";

pub fn require_api_key() -> String {
    match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("Set OPENAI_API_KEY to run this example.");
            process::exit(1);
        }
    }
}

pub fn model() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

pub fn base_url() -> String {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    if key.starts_with("sk-") {
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string())
    } else {
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
    }
}

/// Default client from `OPENAI_*` env vars; not every example uses this helper.
#[allow(dead_code)]
pub fn client_from_env() -> Result<Client, ClientBuildError> {
    Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(base_url())
        .build()
}

/// Usage summary for example output; not every example calls this.
#[allow(dead_code)]
pub fn usage_line(outcome: &CompletionOutcome) -> String {
    match &outcome.usage {
        Some(u) => format!(
            "usage: prompt={} completion={} total={}",
            u.prompt_tokens, u.completion_tokens, u.total_tokens
        ),
        None => "usage: n/a".to_string(),
    }
}

/// Simple JSON tool backed by a Rust closure (used by examples 03–20, not all binaries).
#[allow(dead_code)]
pub struct FnTool<F>
where
    F: Fn(Value) -> Result<Value, String> + Send + Sync,
{
    spec: ToolSpec,
    callback: F,
}

#[allow(dead_code)]
impl<F> FnTool<F>
where
    F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
{
    pub fn new(spec: ToolSpec, callback: F) -> Self {
        Self { spec, callback }
    }

    pub fn into_arc(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

#[async_trait]
#[allow(dead_code)]
impl<F> Tool for FnTool<F>
where
    F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        (self.callback)(arguments)
            .map_err(|message| ToolInvokeError::handler(message, Some(self.spec.name.clone())))
    }
}
