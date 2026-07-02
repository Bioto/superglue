//! Multi-turn conversation benchmark — feeds history forward and measures context growth.

use super::bench_configs::{BenchConfig, ConfigKind};
use super::benchmark_tools;
use super::conversation_profiles::{
    aaak_compression_enabled_multiturn, multiturn_system_prompt, summarize_config_multiturn,
    MULTITURN_MAX_TOOL_ROUNDS, MULTITURN_PROMPTS,
};

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use superglue::chat::{ChatError, ChatOptions, complete_with_tools};
use superglue::context::SummarizeContextConfig;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{ToolMode, ToolRegistry, ROUTER_TOOL_NAME};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Turns that trigger a tool call in the wiremock script (0-based).
const TOOL_TURN_INDICES: &[usize] = &[2, 5, 8, 10];

const COMPLETION_TOKENS_ESTIMATE: u32 = 50;

/// Metrics for a full multi-turn conversation run.
#[derive(Debug, Clone, Default)]
pub struct MultiTurnMetrics {
    pub per_turn_context_msgs: Vec<usize>,
    pub per_turn_context_tokens: Vec<u32>,
    pub final_context_msgs: usize,
    pub final_context_tokens: u32,
    pub peak_context_tokens: u32,
    pub cum_total_tokens: u32,
    pub turn_errors: u32,
}

#[derive(Debug, Default)]
struct TokenCollector {
    cum_total_tokens: u32,
}

struct TurnScriptState {
    calls_in_session: AtomicU32,
    turn_index: AtomicU32,
}

fn approx_context_tokens(messages: &[ChatMessage]) -> u32 {
    serde_json::to_string(messages)
        .map(|s| (s.len() / 4) as u32)
        .unwrap_or(0)
}

fn summarize_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "USR: rate limits 3000 RPM | audit schema | pool timeout_ms=8500 | AAAK encoding"
            },
            "finish_reason": "stop"
        }]
    })
}

fn aaak_compress_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "USR: rate limits 3000 RPM | schema cols | pool timeout | AAAK | cookies Strict"
            },
            "finish_reason": "stop"
        }]
    })
}

fn is_context_compress_request(body: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return false;
    };
    v.get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|arr| {
            arr.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|t| {
                        t.contains("conversation summarizer")
                            || t.contains("expert lossless compressor")
                    })
            })
        })
}

fn is_aaak_compress_request(body: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return false;
    };
    v.get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|arr| {
            arr.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|t| t.contains("Encode in AAAK"))
            })
        })
}

fn router_call_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_router",
                    "type": "function",
                    "function": {
                        "name": ROUTER_TOOL_NAME,
                        "arguments": "{\"query\":\"weather\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn route_resolve_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "[\"get_weather\"]"
            },
            "finish_reason": "stop"
        }]
    })
}

fn weather_tool_call_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_w",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn final_text_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {"role": "assistant", "content": "Here is the answer."},
            "finish_reason": "stop"
        }]
    })
}

fn record_tokens(body: &[u8], collector: &Mutex<TokenCollector>) {
    let prompt_est = (body.len() / 4) as u32;
    let mut guard = collector.lock().unwrap();
    guard.cum_total_tokens += prompt_est + COMPLETION_TOKENS_ESTIMATE;
}

fn scripted_turn_response(
    local_call: u32,
    dynamic: bool,
    tool_turn: bool,
) -> Value {
    if !tool_turn {
        return final_text_response();
    }
    if dynamic {
        return match local_call {
            0 => router_call_response(),
            1 => route_resolve_response(),
            2 => weather_tool_call_response(),
            _ => final_text_response(),
        };
    }
    match local_call {
        0 => weather_tool_call_response(),
        _ => final_text_response(),
    }
}

pub fn chat_options_for_multiturn(cfg: &BenchConfig) -> ChatOptions {
    let summarize_context = if cfg.kind() != ConfigKind::Standard {
        summarize_config_multiturn()
    } else {
        SummarizeContextConfig::default()
    };
    ChatOptions {
        tool_mode: cfg.tool_mode,
        condense_tool_messages: cfg.condense_tool_messages,
        aaak_tool_condensing: cfg.aaak_tool_condensing,
        summarize_context,
        aaak_compression_enabled: aaak_compression_enabled_multiturn(cfg.aaak_tool_condensing),
        system_prompt: Some(multiturn_system_prompt(cfg.aaak_tool_condensing)),
        max_tool_rounds: MULTITURN_MAX_TOOL_ROUNDS,
        ..Default::default()
    }
}

/// Run a multi-turn conversation against wiremock; history is fed forward each turn.
pub async fn run_multiturn_wiremock(cfg: &BenchConfig) -> Result<MultiTurnMetrics, ChatError> {
    let dynamic = cfg.tool_mode == ToolMode::Dynamic;
    let server = MockServer::start().await;
    let turn_state = Arc::new(TurnScriptState {
        calls_in_session: AtomicU32::new(0),
        turn_index: AtomicU32::new(0),
    });
    let turn_state2 = Arc::clone(&turn_state);
    let token_collector = Arc::new(Mutex::new(TokenCollector::default()));
    let token_collector2 = Arc::clone(&token_collector);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            record_tokens(&req.body, &token_collector2);
            if is_aaak_compress_request(&req.body) {
                return ResponseTemplate::new(200).set_body_json(aaak_compress_response());
            }
            if is_context_compress_request(&req.body) {
                return ResponseTemplate::new(200).set_body_json(summarize_response());
            }
            let local = turn_state2.calls_in_session.fetch_add(1, Ordering::SeqCst);
            let turn = turn_state2.turn_index.load(Ordering::SeqCst) as usize;
            let tool_turn = TOOL_TURN_INDICES.contains(&turn);
            let body = scripted_turn_response(local, dynamic, tool_turn);
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let registry = ToolRegistry::new();
    benchmark_tools::register_benchmark_tools(&registry)
        .await
        .expect("register tools");

    let mut opts = chat_options_for_multiturn(cfg);
    opts.base_url = server.uri();
    opts.api_key = secrecy::Secret::new("sk-test".to_string());
    opts.model = "mock".into();
    if dynamic {
        opts.tool_route_model = Some("mock".into());
    }

    let mut history = vec![ChatMessage::text("user", MULTITURN_PROMPTS[0])];
    let mut per_turn_context_msgs = Vec::new();
    let mut per_turn_context_tokens = Vec::new();

    for turn in 0..MULTITURN_PROMPTS.len() {
        turn_state.turn_index.store(turn as u32, Ordering::SeqCst);
        turn_state.calls_in_session.store(0, Ordering::SeqCst);

        per_turn_context_msgs.push(history.len());
        per_turn_context_tokens.push(approx_context_tokens(&history));

        let outcome = complete_with_tools(
            &http,
            &registry,
            &HookRegistry::new(),
            &GuardrailRegistry::new(),
            history,
            &opts,
        )
        .await?;

        history = outcome.messages;
        if turn + 1 < MULTITURN_PROMPTS.len() {
            history.push(ChatMessage::text("user", MULTITURN_PROMPTS[turn + 1]));
        }
    }

    let final_context_msgs = history.len();
    let final_context_tokens = approx_context_tokens(&history);
    let peak_context_tokens = per_turn_context_tokens.iter().copied().max().unwrap_or(0);
    let cum_total_tokens = token_collector.lock().unwrap().cum_total_tokens;

    Ok(MultiTurnMetrics {
        per_turn_context_msgs,
        per_turn_context_tokens,
        final_context_msgs,
        final_context_tokens,
        peak_context_tokens,
        cum_total_tokens,
        turn_errors: 0,
    })
}
