//! Shared wiremock fixtures for context optimization benchmarks and regression tests.

#[path = "bench_configs.rs"]
mod bench_configs;
#[path = "bench_metrics.rs"]
mod bench_metrics;
#[path = "bench_report.rs"]
mod bench_report;
#[path = "benchmark_tools.rs"]
mod benchmark_tools;
#[path = "conversation_profiles.rs"]
pub mod conversation_profiles;
#[path = "multiturn.rs"]
pub mod multiturn;

pub use bench_configs::{BENCH_CONFIGS, BenchConfig};
pub use benchmark_tools::FAT_RAW_CHARS;
pub use conversation_profiles::ConversationProfile;
pub use multiturn::{MultiTurnMetrics, run_multiturn_wiremock};

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use superglue::chat::{ChatError, ChatOptions, CompletionOutcome, complete_with_tools};
use superglue::events::StatusEmitter;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::tools::{CODE_TOOL_NAME, ROUTER_TOOL_NAME, ToolMode, ToolRegistry};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use conversation_profiles::ToolExecutionStyle;

/// Aggregated metrics for one benchmark scenario.
#[derive(Debug, Clone, Default)]
pub struct ScenarioMetrics {
    pub llm_rounds: u32,
    pub tool_calls: u32,
    pub peak_context_msgs: usize,
    pub peak_tools: usize,
    pub cum_prompt_tokens: u32,
    pub cum_total_tokens: u32,
    pub history_bytes: usize,
    pub completed: bool,
}

#[derive(Debug, Default)]
struct RequestMetricsCollector {
    peak_tools: usize,
    peak_context_msgs: usize,
    cum_prompt_tokens: u32,
    cum_total_tokens: u32,
}

const COMPLETION_TOKENS_ESTIMATE: u32 = 50;

fn summarize_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "USR: trip planning context compressed."
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
                "content": "USR: weather=72F | flights=NYC-Paris | rate=0.92"
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
                        "arguments": "{\"query\":\"trip planning tools\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn route_resolve_response(tools: &[&str]) -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": serde_json::to_string(tools).unwrap_or_else(|_| "[]".into())
            },
            "finish_reason": "stop"
        }]
    })
}

fn tool_call_response(tool_name: &str, idx: u32) -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn multi_tool_call_response(tools: &[&str]) -> Value {
    let tool_calls: Vec<Value> = tools
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            json!({
                "id": format!("call_{idx}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": "{}"
                }
            })
        })
        .collect();
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": tool_calls
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn final_text_response() -> Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {"role": "assistant", "content": "Trip plan complete."},
            "finish_reason": "stop"
        }]
    })
}

fn code_tool_call_response(tools: &[&str]) -> Value {
    let mut source = String::new();
    for name in tools {
        source.push_str(&format!("const {name} = tools.{name}({{}});\n"));
    }
    source.push_str("return {\n");
    source
        .push_str("  temp: (typeof get_weather !== 'undefined' && get_weather.temp_f) || null,\n");
    source.push_str(
        "  highs: (typeof get_forecast !== 'undefined' && get_forecast.days) ? get_forecast.days.length : 0,\n",
    );
    source.push_str("  avg: (typeof calculate !== 'undefined' && calculate.result) || null,\n");
    source.push_str(
        "  price: (typeof search_flights !== 'undefined' && search_flights.price_usd) || null,\n",
    );
    source.push_str(
        "  rate: (typeof get_exchange_rate !== 'undefined' && get_exchange_rate.rate) || null,\n",
    );
    source.push_str(
        "  hello: (typeof translate_text !== 'undefined' && translate_text.translation) || null\n",
    );
    source.push_str("};\n");
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_code",
                    "type": "function",
                    "function": {
                        "name": CODE_TOOL_NAME,
                        "arguments": serde_json::to_string(&json!({ "source": source })).unwrap()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn scripted_response(call_index: u32, mode: ToolMode, profile: ConversationProfile) -> Value {
    let expected = profile.expected_tools();
    let uses_router = mode.uses_router();
    let tool_offset = if uses_router { 2 } else { 0 };
    let parallel = profile.execution_style() == ToolExecutionStyle::Parallel;

    if mode == ToolMode::Code {
        return match call_index {
            0 => router_call_response(),
            1 => route_resolve_response(expected),
            2 => code_tool_call_response(expected),
            _ => final_text_response(),
        };
    }

    if uses_router {
        match call_index {
            0 => return router_call_response(),
            1 => return route_resolve_response(expected),
            _ => {
                let ti = call_index.saturating_sub(tool_offset);
                if parallel {
                    return if ti == 0 {
                        multi_tool_call_response(expected)
                    } else {
                        final_text_response()
                    };
                }
                if (ti as usize) < expected.len() {
                    return tool_call_response(expected[ti as usize], ti);
                }
                return final_text_response();
            }
        }
    }

    if parallel {
        return if call_index == 0 {
            multi_tool_call_response(expected)
        } else {
            final_text_response()
        };
    }

    if (call_index as usize) < expected.len() {
        tool_call_response(expected[call_index as usize], call_index)
    } else {
        final_text_response()
    }
}

fn record_request(body: &[u8], collector: &Mutex<RequestMetricsCollector>) {
    let Ok(parsed) = serde_json::from_slice::<Value>(body) else {
        return;
    };
    let mut guard = collector.lock().unwrap();
    let prompt_est = (body.len() / 4) as u32;
    guard.cum_prompt_tokens += prompt_est;
    guard.cum_total_tokens += prompt_est + COMPLETION_TOKENS_ESTIMATE;
    if let Some(msgs) = parsed.get("messages").and_then(|m| m.as_array()) {
        guard.peak_context_msgs = guard.peak_context_msgs.max(msgs.len());
    }
    if let Some(tools) = parsed.get("tools").and_then(|t| t.as_array()) {
        guard.peak_tools = guard.peak_tools.max(tools.len());
    }
}

async fn register_tools_with_raw(registry: &ToolRegistry, raw_chars: usize) {
    if raw_chars == 0 {
        benchmark_tools::register_benchmark_tools(registry)
            .await
            .expect("register benchmark tools");
    } else {
        benchmark_tools::register_fat_benchmark_tools(registry, raw_chars)
            .await
            .expect("register fat benchmark tools");
    }
}

/// Run one scripted multi-round tool-chain scenario against wiremock and collect metrics.
pub async fn run_scenario(
    opts: ChatOptions,
    profile: ConversationProfile,
) -> Result<(CompletionOutcome, ScenarioMetrics), ChatError> {
    run_scenario_with_raw(opts, profile, 0).await
}

/// Same as [`run_scenario`] but each tool result includes a `raw` blob of `raw_chars`.
pub async fn run_scenario_with_raw(
    opts: ChatOptions,
    profile: ConversationProfile,
    raw_chars: usize,
) -> Result<(CompletionOutcome, ScenarioMetrics), ChatError> {
    let mode = opts.tool_mode;
    let uses_router = mode.uses_router();
    let initial_messages = if raw_chars > 0 {
        profile.fat_initial_messages()
    } else {
        profile.initial_messages()
    };
    let server = MockServer::start().await;
    let call_idx = Arc::new(AtomicU32::new(0));
    let call_idx2 = Arc::clone(&call_idx);
    let collector = Arc::new(Mutex::new(RequestMetricsCollector::default()));
    let collector2 = Arc::clone(&collector);

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            record_request(&req.body, &collector2);
            if is_aaak_compress_request(&req.body) {
                return ResponseTemplate::new(200).set_body_json(aaak_compress_response());
            }
            if is_context_compress_request(&req.body) {
                return ResponseTemplate::new(200).set_body_json(summarize_response());
            }
            let i = call_idx2.fetch_add(1, Ordering::SeqCst);
            let body = scripted_response(i, mode, profile);
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let registry = ToolRegistry::new();
    register_tools_with_raw(&registry, raw_chars).await;

    let mut opts = opts;
    opts.base_url = server.uri();
    opts.api_key = secrecy::SecretString::from("sk-test".to_string());
    opts.model = "mock".into();
    opts.max_tool_rounds = 12;
    if raw_chars > 0 {
        opts.tool_result_max_chars = raw_chars.saturating_add(4_096);
    }
    let tool_call_collector = Arc::new(bench_metrics::ToolCallCollector::new());
    let status_emitter = StatusEmitter::new();
    status_emitter
        .subscribe(Arc::clone(&tool_call_collector) as Arc<dyn superglue::events::StatusSubscriber>)
        .await;
    opts.status_emitter = Some(Arc::new(status_emitter));
    if uses_router {
        opts.tool_route_model = Some("mock".into());
    }

    let outcome = complete_with_tools(
        &http,
        &registry,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        initial_messages,
        &opts,
    )
    .await?;

    let history_bytes = serde_json::to_string(&outcome.messages)
        .map(|s| s.len())
        .unwrap_or(0);
    let collected = collector.lock().unwrap();
    let tool_calls = tool_call_collector.count();
    let completed = bench_metrics::completed_expected(&outcome.messages, profile.expected_tools());

    Ok((
        outcome.clone(),
        ScenarioMetrics {
            llm_rounds: outcome.rounds,
            tool_calls,
            peak_context_msgs: collected.peak_context_msgs,
            peak_tools: collected.peak_tools,
            cum_prompt_tokens: collected.cum_prompt_tokens,
            cum_total_tokens: collected.cum_total_tokens,
            history_bytes,
            completed,
        },
    ))
}

pub fn chat_options_from_bench(cfg: &BenchConfig) -> ChatOptions {
    ChatOptions {
        tool_mode: cfg.tool_mode,
        condense_tool_messages: cfg.condense_tool_messages,
        aaak_tool_condensing: cfg.aaak_tool_condensing,
        ..Default::default()
    }
}

pub fn chat_options_for_run(cfg: &BenchConfig, profile: ConversationProfile) -> ChatOptions {
    let mut opts = chat_options_from_bench(cfg);
    opts.summarize_context = profile.summarize_config();
    opts.aaak_compression_enabled = profile.aaak_compression_enabled(cfg.aaak_tool_condensing);
    opts.system_prompt = Some(conversation_profiles::benchmark_system_prompt(
        cfg.aaak_tool_condensing,
        cfg.tool_mode,
    ));
    opts
}

pub fn chat_options_for_fat_run(cfg: &BenchConfig, profile: ConversationProfile) -> ChatOptions {
    let mut opts = chat_options_for_run(cfg, profile);
    opts.system_prompt = Some(conversation_profiles::fat_benchmark_system_prompt(
        cfg.aaak_tool_condensing,
        cfg.tool_mode,
    ));
    opts.tool_result_max_chars = FAT_RAW_CHARS.saturating_add(4_096);
    opts
}

fn scenario_to_row(label: String, m: &ScenarioMetrics) -> bench_report::BenchmarkRow {
    bench_report::BenchmarkRow {
        label,
        llm_rounds: m.llm_rounds,
        tool_calls: m.tool_calls,
        peak_context_msgs: m.peak_context_msgs,
        peak_tools: m.peak_tools,
        cum_prompt_tokens: m.cum_prompt_tokens,
        cum_total_tokens: m.cum_total_tokens,
        history_bytes: m.history_bytes,
        completed: m.completed,
        error: None,
    }
}

#[cfg(not(test))]
pub fn print_comparison_table(rows: &[(String, ScenarioMetrics)]) {
    let standard_total = rows
        .iter()
        .find(|(label, _)| label.ends_with("_standard"))
        .map(|(_, m)| m.cum_total_tokens)
        .unwrap_or(0);

    let report_rows: Vec<bench_report::BenchmarkRow> = rows
        .iter()
        .map(|(label, m)| scenario_to_row(label.clone(), m))
        .collect();

    bench_report::print_benchmark_table(&report_rows, standard_total);
}

pub fn print_multiturn_comparison(rows: &[(String, MultiTurnMetrics)]) {
    let report_rows: Vec<(String, bench_report::MultiTurnReport)> = rows
        .iter()
        .map(|(label, m)| {
            (
                label.clone(),
                bench_report::MultiTurnReport {
                    final_context_msgs: m.final_context_msgs,
                    final_context_tokens: m.final_context_tokens,
                    peak_context_tokens: m.peak_context_tokens,
                    cum_total_tokens: m.cum_total_tokens,
                    per_turn_context_msgs: m.per_turn_context_msgs.clone(),
                    turn_errors: m.turn_errors,
                },
            )
        })
        .collect();
    bench_report::print_multiturn_table(&report_rows);
}
