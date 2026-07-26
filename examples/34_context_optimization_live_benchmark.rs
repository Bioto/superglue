//! Example 34: Live context optimization benchmark — GlueLLM-style tool chains + multi-turn context growth.
//!
//! Requires `OPENAI_API_KEY` and a valid `OPENAI_MODEL`.

mod support;

#[path = "../benches/support/bench_configs.rs"]
mod bench_configs;
#[path = "../benches/support/bench_metrics.rs"]
mod bench_metrics;
#[path = "../benches/support/bench_report.rs"]
mod bench_report;
#[path = "../benches/support/conversation_profiles.rs"]
mod conversation_profiles;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use bench_configs::{BENCH_CONFIGS, ConfigKind};
use bench_report::{
    BenchmarkRow, LIVE_BENCHMARK_NOTE, MultiTurnReport, print_benchmark_table,
    print_multiturn_table,
};
use conversation_profiles::aaak_compression_enabled_multiturn;
use conversation_profiles::{
    ConversationProfile, MULTITURN_MAX_TOOL_ROUNDS, MULTITURN_PROMPTS, benchmark_system_prompt,
    multiturn_system_prompt, summarize_config_multiturn,
};
use superglue::events::{ProcessEvent, ProcessEventKind, StatusEmitter, StatusSubscriber};
use superglue::openai::ChatMessage;
use support::context_tools::register_context_optimization_tools;
use support::{base_url, model, require_api_key};

struct CumulativeTokenCollector {
    prompt: AtomicU32,
    total: AtomicU32,
}

impl CumulativeTokenCollector {
    fn new() -> Self {
        Self {
            prompt: AtomicU32::new(0),
            total: AtomicU32::new(0),
        }
    }

    fn prompt(&self) -> u32 {
        self.prompt.load(Ordering::Relaxed)
    }

    fn total(&self) -> u32 {
        self.total.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl StatusSubscriber for CumulativeTokenCollector {
    async fn on_event(&self, event: ProcessEvent) {
        if event.kind == ProcessEventKind::LlmCallEnd {
            if let Some(usage) = event.usage {
                self.prompt
                    .fetch_add(usage.prompt_tokens, Ordering::Relaxed);
                self.total.fetch_add(usage.total_tokens, Ordering::Relaxed);
            }
        }
    }
}

fn approx_context_tokens(messages: &[ChatMessage]) -> u32 {
    serde_json::to_string(messages)
        .map(|s| (s.len() / 4) as u32)
        .unwrap_or(0)
}

async fn run_live_config(
    api_key: &str,
    model: &str,
    base_url: &str,
    profile: ConversationProfile,
    cfg: &bench_configs::BenchConfig,
) -> BenchmarkRow {
    let label = format!("{}_{}", profile.name(), cfg.label);
    let emitter = Arc::new(StatusEmitter::new());
    let token_collector = Arc::new(CumulativeTokenCollector::new());
    emitter
        .subscribe(Arc::clone(&token_collector) as Arc<dyn StatusSubscriber>)
        .await;

    let client = match superglue::Client::builder()
        .api_key(api_key)
        .model(model)
        .base_url(base_url)
        .system_prompt(benchmark_system_prompt(cfg.aaak_tool_condensing))
        .tool_mode(cfg.tool_mode)
        .condense_tool_messages(cfg.condense_tool_messages)
        .aaak_tool_condensing(cfg.aaak_tool_condensing)
        .summarize_context(profile.summarize_config())
        .aaak_compression_enabled(profile.aaak_compression_enabled(cfg.aaak_tool_condensing))
        .max_tool_rounds(12)
        .status_emitter(emitter)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return BenchmarkRow {
                label,
                llm_rounds: 0,
                tool_calls: 0,
                peak_context_msgs: 0,
                peak_tools: cfg.expected_peak_tools(profile),
                cum_prompt_tokens: 0,
                cum_total_tokens: 0,
                history_bytes: 0,
                completed: false,
                error: Some(format!("build:{e}")),
            };
        }
    };

    if let Err(e) = register_context_optimization_tools(&client).await {
        return BenchmarkRow {
            label,
            llm_rounds: 0,
            tool_calls: 0,
            peak_context_msgs: 0,
            peak_tools: cfg.expected_peak_tools(profile),
            cum_prompt_tokens: 0,
            cum_total_tokens: 0,
            history_bytes: 0,
            completed: false,
            error: Some(format!("tools:{e}")),
        };
    }

    let messages = profile.initial_messages();
    match client
        .complete_messages(messages, superglue::CallOptions::default())
        .await
    {
        Ok(outcome) => {
            let history_bytes = serde_json::to_string(&outcome.messages)
                .map(|s| s.len())
                .unwrap_or(0);
            let tool_calls = bench_metrics::count_tool_calls(&outcome.messages);
            let completed =
                bench_metrics::completed_expected(&outcome.messages, profile.expected_tools());
            let cum_total = token_collector.total();
            let cum_prompt = token_collector.prompt();
            eprintln!(
                "  {label} → {} rounds, {} tool_calls, {} cum_total, completed={completed}",
                outcome.rounds, tool_calls, cum_total
            );
            BenchmarkRow {
                label,
                llm_rounds: outcome.rounds,
                tool_calls,
                peak_context_msgs: outcome.messages.len(),
                peak_tools: cfg.expected_peak_tools(profile),
                cum_prompt_tokens: cum_prompt,
                cum_total_tokens: cum_total,
                history_bytes,
                completed,
                error: None,
            }
        }
        Err(e) => {
            eprintln!("  {label} → ERROR: {e}");
            BenchmarkRow {
                label,
                llm_rounds: 0,
                tool_calls: 0,
                peak_context_msgs: 0,
                peak_tools: cfg.expected_peak_tools(profile),
                cum_prompt_tokens: 0,
                cum_total_tokens: 0,
                history_bytes: 0,
                completed: false,
                error: Some(e.to_string()),
            }
        }
    }
}

async fn run_live_multiturn(
    api_key: &str,
    model: &str,
    base_url: &str,
    cfg: &bench_configs::BenchConfig,
) -> MultiTurnReport {
    let label = format!("multiturn_{}", cfg.label);
    let emitter = Arc::new(StatusEmitter::new());
    let token_collector = Arc::new(CumulativeTokenCollector::new());
    emitter
        .subscribe(Arc::clone(&token_collector) as Arc<dyn StatusSubscriber>)
        .await;

    let summarize_context = if cfg.kind() != ConfigKind::Standard {
        summarize_config_multiturn()
    } else {
        superglue::context::SummarizeContextConfig::default()
    };

    let client = match superglue::Client::builder()
        .api_key(api_key)
        .model(model)
        .base_url(base_url)
        .system_prompt(multiturn_system_prompt(cfg.aaak_tool_condensing))
        .tool_mode(cfg.tool_mode)
        .condense_tool_messages(cfg.condense_tool_messages)
        .aaak_tool_condensing(cfg.aaak_tool_condensing)
        .summarize_context(summarize_context)
        .aaak_compression_enabled(aaak_compression_enabled_multiturn(cfg.aaak_tool_condensing))
        .max_tool_rounds(MULTITURN_MAX_TOOL_ROUNDS)
        .status_emitter(emitter)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  {label} → build ERROR: {e}");
            return MultiTurnReport {
                final_context_msgs: 0,
                final_context_tokens: 0,
                peak_context_tokens: 0,
                cum_total_tokens: 0,
                per_turn_context_msgs: Vec::new(),
                turn_errors: 0,
            };
        }
    };

    if let Err(e) = register_context_optimization_tools(&client).await {
        eprintln!("  {label} → tools ERROR: {e}");
        return MultiTurnReport {
            final_context_msgs: 0,
            final_context_tokens: 0,
            peak_context_tokens: 0,
            cum_total_tokens: 0,
            per_turn_context_msgs: Vec::new(),
            turn_errors: 0,
        };
    }

    let mut history = vec![ChatMessage::text("user", MULTITURN_PROMPTS[0])];
    let mut per_turn_context_msgs = Vec::new();
    let mut per_turn_context_tokens = Vec::new();
    let mut turn_errors = 0u32;

    for turn in 0..MULTITURN_PROMPTS.len() {
        per_turn_context_msgs.push(history.len());
        per_turn_context_tokens.push(approx_context_tokens(&history));

        let history_before = history.clone();
        match client
            .complete_messages(history.clone(), superglue::CallOptions::default())
            .await
        {
            Ok(outcome) => {
                history = outcome.messages;
                if turn + 1 < MULTITURN_PROMPTS.len() {
                    history.push(ChatMessage::text("user", MULTITURN_PROMPTS[turn + 1]));
                }
            }
            Err(e) => {
                turn_errors += 1;
                eprintln!("  {label} turn {turn} → ERROR: {e}");
                history = history_before;
                if turn + 1 < MULTITURN_PROMPTS.len() {
                    history.push(ChatMessage::text("user", MULTITURN_PROMPTS[turn + 1]));
                }
            }
        }
    }

    let final_context_tokens = approx_context_tokens(&history);
    let peak_context_tokens = per_turn_context_tokens.iter().copied().max().unwrap_or(0);
    eprintln!(
        "  {label} → final {} msgs, {} ctx_tokens, {} cum_total, errors={turn_errors}",
        history.len(),
        final_context_tokens,
        token_collector.total(),
    );

    MultiTurnReport {
        final_context_msgs: history.len(),
        final_context_tokens,
        peak_context_tokens,
        cum_total_tokens: token_collector.total(),
        per_turn_context_msgs,
        turn_errors,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = require_api_key();
    let model = model();
    let base_url = base_url();

    println!("{LIVE_BENCHMARK_NOTE}\n");
    eprintln!("Live benchmark: model={model} base_url={base_url}");
    eprintln!(
        "Running {} configs × {} profiles…\n",
        BENCH_CONFIGS.len(),
        ConversationProfile::ALL.len()
    );

    for profile in ConversationProfile::ALL {
        println!("=== {} scenario ===\n", profile.name());
        let mut rows: Vec<BenchmarkRow> = Vec::new();

        for cfg in BENCH_CONFIGS {
            let row = run_live_config(&api_key, &model, &base_url, *profile, cfg).await;
            rows.push(row);
        }

        let standard_total = rows
            .iter()
            .find(|r| r.label.ends_with("_standard") && r.error.is_none())
            .map(|r| r.cum_total_tokens)
            .unwrap_or(0);

        print_benchmark_table(&rows, standard_total);
        println!();
    }

    println!("=== multi-turn conversation (context growth) ===\n");
    let mut multiturn_rows: Vec<(String, MultiTurnReport)> = Vec::new();
    for cfg in BENCH_CONFIGS {
        let report = run_live_multiturn(&api_key, &model, &base_url, cfg).await;
        multiturn_rows.push((format!("multiturn_{}", cfg.label), report));
    }
    print_multiturn_table(&multiturn_rows);
    println!();

    Ok(())
}
