//! JSON config parsing (matches superglue-js constructor defaults).

use std::num::NonZeroU32;
use std::sync::Mutex;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use superglue::agents::AgentSpec;
use superglue::client::{bootstrap_from_parts, provider_id_from_str, BindingBootstrapConfig};
use superglue::fallback::ModelFallbackChain;
use superglue::http::{ClientConfig, HttpClient, RetryPolicy};
use superglue::providers::ProviderCredentials;

use crate::handle::ClientState;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCreateJson {
    pub api_key: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    pub system_prompt: Option<String>,
    #[serde(default = "default_u16_16")]
    pub max_tool_rounds: u32,
    #[serde(default = "default_u16_3")]
    pub max_retries: u32,
    #[serde(default = "default_u16_50")]
    pub retry_initial_delay_ms: u32,
    #[serde(default = "default_u16_2000")]
    pub retry_max_delay_ms: u32,
    #[serde(default = "default_f_2")]
    pub retry_multiplier: f64,
    pub requests_per_second: Option<u32>,
    #[serde(default = "default_i64_60")]
    pub timeout_secs: i64,
    #[serde(default = "default_i64_30")]
    pub connect_timeout_secs: i64,
    #[serde(default = "default_u16_3")]
    pub max_output_retries: u32,
    #[serde(default = "default_u16_50_pool")]
    pub pool_max_idle_per_host: u32,
    pub pool_idle_timeout_secs: Option<i64>,
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub model_fallback_models: Option<Vec<String>>,
    #[serde(default)]
    pub api_keys: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub requests_per_second_for: Option<std::collections::HashMap<String, u32>>,
    pub max_upload_bytes: Option<u32>,
    #[serde(default = "default_tool_mode")]
    pub tool_mode: String,
    pub tool_route_model: Option<String>,
    #[serde(default)]
    pub condense_tool_messages: bool,
    #[serde(default)]
    pub aaak_tool_condensing: bool,
    #[serde(default)]
    pub summarize_context_enabled: bool,
    #[serde(default = "default_u16_20")]
    pub summarize_context_threshold: u32,
    #[serde(default = "default_u16_6")]
    pub summarize_context_keep_recent: u32,
    #[serde(default)]
    pub aaak_compression_enabled: bool,
    pub aaak_compression_model: Option<String>,
}

fn default_model() -> String {
    "gpt-5.4-nano-2026-03-17-mini".to_string()
}
fn default_base_url() -> String {
    "https://api.openai.com".to_string()
}
fn default_u16_16() -> u32 {
    16
}
fn default_u16_3() -> u32 {
    3
}
fn default_u16_50() -> u32 {
    50
}
fn default_u16_2000() -> u32 {
    2000
}
fn default_f_2() -> f64 {
    2.0
}
fn default_i64_60() -> i64 {
    60
}
fn default_i64_30() -> i64 {
    30
}
fn default_u16_50_pool() -> u32 {
    50
}
fn default_u16_20() -> u32 {
    20
}
fn default_u16_6() -> u32 {
    6
}
fn default_tool_mode() -> String {
    "standard".to_string()
}

/// Parse and build [`ClientState`] (same as `superglue-js` `Client::new`).
pub fn build_client_state(json: &str) -> Result<ClientState, String> {
    let c: ClientCreateJson =
        serde_json::from_str(json).map_err(|e| format!("client config: {e}"))?;

    let provider_creds = c.api_keys.filter(|m| !m.is_empty()).map(|m| {
        let mut creds = ProviderCredentials::from_env();
        for (name, key) in m {
            if let Some(pid) = provider_id_from_str(&name) {
                creds.insert_key(pid, key);
            }
        }
        creds
    });
    let mut provider_qps = std::collections::HashMap::new();
    if let Some(map) = &c.requests_per_second_for {
        for (name, qps) in map {
            if let Some(pid) = provider_id_from_str(name) {
                provider_qps.insert(pid, *qps);
            }
        }
    }
    let model_fallback = c.model_fallback_models.and_then(|models| {
        if models.is_empty() {
            None
        } else {
            Some(ModelFallbackChain::new(models))
        }
    });

    let bootstrap = bootstrap_from_parts(BindingBootstrapConfig {
        api_key: c.api_key,
        base_url: c.base_url,
        model: c.model,
        system_prompt: c.system_prompt,
        max_tool_rounds: c.max_tool_rounds,
        max_retries: c.max_retries,
        retry_initial_delay_ms: u64::from(c.retry_initial_delay_ms),
        retry_max_delay_ms: u64::from(c.retry_max_delay_ms),
        retry_multiplier: c.retry_multiplier,
        requests_per_second: c.requests_per_second,
        timeout: Duration::from_secs(c.timeout_secs.max(0) as u64),
        connect_timeout: Duration::from_secs(c.connect_timeout_secs.max(0) as u64),
        max_output_retries: c.max_output_retries,
        pool_max_idle_per_host: c.pool_max_idle_per_host as usize,
        pool_idle_timeout: c
            .pool_idle_timeout_secs
            .map(|s| Duration::from_secs(s.max(0) as u64)),
        reasoning_effort: c.reasoning_effort.clone(),
        status_emitter: None,
        model_fallback,
        provider_credentials: provider_creds,
        provider_qps,
        max_upload_bytes: c
            .max_upload_bytes
            .map(|b| b as usize)
            .unwrap_or_else(superglue::files::default_max_upload_bytes),
    })
    .map_err(|e| e.to_string())?;

    let mut options = bootstrap.options;
    options.tool_mode = match c.tool_mode.as_str() {
        "dynamic" => superglue::tools::ToolMode::Dynamic,
        _ => superglue::tools::ToolMode::Standard,
    };
    options.tool_route_model = c.tool_route_model.clone();
    options.condense_tool_messages = c.condense_tool_messages;
    options.aaak_tool_condensing = c.aaak_tool_condensing;
    options.summarize_context = superglue::context::SummarizeContextConfig {
        enabled: c.summarize_context_enabled,
        threshold: c.summarize_context_threshold as usize,
        keep_recent: c.summarize_context_keep_recent as usize,
    };
    options.aaak_compression_enabled = c.aaak_compression_enabled;
    options.aaak_compression_model = c.aaak_compression_model.clone();

    use superglue::guardrails::GuardrailRegistry;
    use superglue::hooks::HookRegistry;
    use superglue::tools::ToolRegistry;
    use tokio::sync::watch;

    let (cancel_tx, _cancel_rx) = watch::channel(0_u64);

    Ok(ClientState {
        options,
        registry: std::sync::Arc::new(ToolRegistry::new()),
        hooks: std::sync::Arc::new(HookRegistry::new()),
        guardrails: std::sync::Arc::new(
            GuardrailRegistry::new().with_max_output_retries(c.max_output_retries),
        ),
        http: std::sync::Arc::new(bootstrap.http),
        max_upload_bytes: bootstrap.max_upload_bytes,
        cancel_tx,
        mcp_sessions: Mutex::new(Vec::new()),
    })
}

/// Attach a [`StatusEmitter`] handle to client options after JSON parse.
pub fn apply_status_emitter(
    state: &mut ClientState,
    emitter: std::sync::Arc<superglue::events::StatusEmitter>,
) {
    state.options.status_emitter = Some(emitter);
}


#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSpecJson {
    pub name: String,
    pub persona: String,
    pub goals: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_u16_16")]
    pub max_tool_rounds: u32,
    #[serde(default = "default_u16_3")]
    pub max_output_retries: u32,
    pub system_prompt: Option<String>,
    pub reasoning_effort: Option<String>,
}

pub fn parse_agent_spec(json: &str) -> Result<AgentSpec, String> {
    let a: AgentSpecJson = serde_json::from_str(json).map_err(|e| format!("agent spec: {e}"))?;
    let mut spec = AgentSpec::new(a.name, a.persona)
        .with_model(a.model)
        .with_max_tool_rounds(a.max_tool_rounds);
    spec.max_output_retries = a.max_output_retries;
    if let Some(g) = a.goals {
        spec.goals = g;
    }
    if let Some(c) = a.constraints {
        spec.constraints = c;
    }
    if let Some(sp) = a.system_prompt {
        spec = spec.with_system_prompt(sp);
    }
    spec.reasoning_effort = a.reasoning_effort;
    Ok(spec)
}

// --- HTTP-only config for `AgentEngine` (mirrors `JsAgentEngine::new` HTTP args) ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineHttpJson {
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: Option<String>,
    #[serde(default = "default_u16_3")]
    pub max_retries: u32,
    #[serde(default = "default_u16_50")]
    pub retry_initial_delay_ms: u32,
    #[serde(default = "default_u16_2000")]
    pub retry_max_delay_ms: u32,
    #[serde(default = "default_f_2")]
    pub retry_multiplier: f64,
    pub requests_per_second: Option<u32>,
    #[serde(default = "default_i64_60")]
    pub timeout_secs: i64,
    #[serde(default = "default_i64_30")]
    pub connect_timeout_secs: i64,
    #[serde(default = "default_u16_50_pool")]
    pub pool_max_idle_per_host: u32,
    pub pool_idle_timeout_secs: Option<i64>,
}

use crate::handle::EngineState;
use std::sync::Arc;

/// Build `EngineState` from agent spec JSON + http JSON (same as `superglue-js` `AgentEngine`).
pub fn build_engine_state(spec_json: &str, http_json: &str) -> Result<EngineState, String> {
    let spec = parse_agent_spec(spec_json)?;
    let h: EngineHttpJson =
        serde_json::from_str(http_json).map_err(|e| format!("engine http: {e}"))?;
    let cfg = ClientConfig {
        retry: RetryPolicy {
            max_retries: h.max_retries,
            initial_interval_ms: u64::from(h.retry_initial_delay_ms),
            max_interval_ms: u64::from(h.retry_max_delay_ms),
            multiplier: h.retry_multiplier,
        },
        quota_per_second: h.requests_per_second.and_then(NonZeroU32::new),
        timeout: Duration::from_secs(h.timeout_secs.max(0) as u64),
        connect_timeout: Duration::from_secs(h.connect_timeout_secs.max(0) as u64),
        pool_max_idle_per_host: h.pool_max_idle_per_host as usize,
        pool_idle_timeout: h
            .pool_idle_timeout_secs
            .map(|s| Duration::from_secs(s.max(0) as u64)),
        ..ClientConfig::default()
    };
    let http = HttpClient::new(cfg).map_err(|e| e.to_string())?;
    use superglue::chat::ChatOptions;
    use superglue::guardrails::GuardrailRegistry;
    use superglue::hooks::HookRegistry;
    use superglue::tools::ToolRegistry;

    let effective_model = if spec.model.is_empty() {
        h.model
            .unwrap_or_else(|| "gpt-5.4-nano-2026-03-17-mini".to_string())
    } else {
        spec.model.clone()
    };
    let guardrails = Arc::new(
        GuardrailRegistry::new().with_max_output_retries(spec.max_output_retries),
    );
    Ok(EngineState {
        spec: spec.clone(),
        base_options: ChatOptions::new(
            h.base_url.as_deref().unwrap_or("https://api.openai.com"),
            h.api_key,
            effective_model,
        ),
        registry: Arc::new(ToolRegistry::new()),
        hooks: Arc::new(HookRegistry::new()),
        guardrails,
        http: Arc::new(http),
    })
}

// --- JSON value parsing (e.g. batch requests) ---

pub fn value_from_str(s: &str) -> Result<Value, String> {
    serde_json::from_str(s).map_err(|e| e.to_string())
}
