//! Opaque handle table for client / conversation / engine native pointers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use superglue::agents::AgentSpec;
use superglue::events::StatusEmitter;
use superglue::http::HttpClient;
use superglue::openai::ChatMessage;
use superglue::tools::ToolRegistry;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::chat::ChatOptions;
use tokio::sync::watch;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static HANDLES: OnceLock<Mutex<HashMap<u64, KHandle>>> = OnceLock::new();

fn table() -> &'static Mutex<HashMap<u64, KHandle>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct ClientState {
    pub options: ChatOptions,
    pub registry: Arc<ToolRegistry>,
    pub hooks: Arc<HookRegistry>,
    pub guardrails: Arc<GuardrailRegistry>,
    pub http: Arc<HttpClient>,
    pub max_upload_bytes: usize,
    pub cancel_tx: watch::Sender<u64>,
    pub mcp_sessions: Mutex<Vec<Arc<superglue::mcp::McpSession>>>,
}

pub struct EngineState {
    pub spec: AgentSpec,
    pub base_options: ChatOptions,
    pub registry: Arc<ToolRegistry>,
    pub hooks: Arc<HookRegistry>,
    pub guardrails: Arc<GuardrailRegistry>,
    pub http: Arc<HttpClient>,
}

pub struct ConversationState {
    pub state: Arc<ClientState>,
    pub turns: Arc<Mutex<Vec<ChatMessage>>>,
}

pub enum KHandle {
    Client(Arc<ClientState>),
    Conversation(ConversationState),
    Engine(Arc<EngineState>),
    StatusEmitter(Arc<StatusEmitter>),
}

pub fn alloc(h: KHandle) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut t = table().lock().expect("handles");
    t.insert(id, h);
    id
}

pub fn get_client(id: u64) -> Result<Arc<ClientState>, String> {
    let t = table().lock().expect("handles");
    match t.get(&id) {
        Some(KHandle::Client(s)) => Ok(Arc::clone(s)),
        _ => Err(format!("invalid client handle: {id}")),
    }
}

pub fn get_convo(id: u64) -> Result<ConversationState, String> {
    let t = table().lock().expect("handles");
    match t.get(&id) {
        Some(KHandle::Conversation(s)) => Ok(ConversationState {
            state: Arc::clone(&s.state),
            turns: Arc::clone(&s.turns),
        }),
        _ => Err(format!("invalid conversation handle: {id}")),
    }
}

pub fn get_engine(id: u64) -> Result<Arc<EngineState>, String> {
    let t = table().lock().expect("handles");
    match t.get(&id) {
        Some(KHandle::Engine(s)) => Ok(Arc::clone(s)),
        _ => Err(format!("invalid engine handle: {id}")),
    }
}

pub fn get_status_emitter(id: u64) -> Result<Arc<StatusEmitter>, String> {
    let t = table().lock().expect("handles");
    match t.get(&id) {
        Some(KHandle::StatusEmitter(s)) => Ok(Arc::clone(s)),
        _ => Err(format!("invalid status emitter handle: {id}")),
    }
}

/// Remove a handle. Engine / client can also drop registered callback GlobalRefs
/// by dropping the underlying `Arc<ClientState>` (tools hold refs until registry drops).
pub fn remove(id: u64) -> bool {
    let mut t = table().lock().expect("handles");
    t.remove(&id).is_some()
}
