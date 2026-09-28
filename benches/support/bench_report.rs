//! Shared benchmark table formatting for wiremock and live runs.

use super::bench_configs::{self, BENCH_CONFIGS, BenchConfig, ConfigKind};
use super::conversation_profiles::ConversationProfile;

/// One row in a benchmark comparison table.
#[derive(Debug, Clone)]
pub struct BenchmarkRow {
    pub label: String,
    pub llm_rounds: u32,
    pub tool_calls: u32,
    pub peak_context_msgs: usize,
    pub peak_tools: usize,
    pub cum_prompt_tokens: u32,
    pub cum_total_tokens: u32,
    #[allow(dead_code)]
    pub history_bytes: usize,
    pub completed: bool,
    pub error: Option<String>,
}

fn config_for_label(row_label: &str) -> Option<&'static BenchConfig> {
    let cfg_label = row_label
        .split_once('_')
        .map(|(_, rest)| rest)
        .unwrap_or(row_label);
    // Profile names contain underscores (short_chain, long_chain).
    for cfg in BENCH_CONFIGS {
        if row_label.ends_with(&format!("_{}", cfg.label)) {
            return Some(cfg);
        }
    }
    let _ = cfg_label;
    None
}

fn profile_for_label(row_label: &str) -> Option<ConversationProfile> {
    ConversationProfile::ALL
        .iter()
        .copied()
        .find(|p| row_label.starts_with(&format!("{}_", p.name())))
}

/// Printed once at the start of live benchmarks.
pub const LIVE_BENCHMARK_NOTE: &str = "Note: Three benchmark sections — (1) single-turn tool chains: headline \
metric is cumulative total tokens across all LLM rounds (GlueLLM parity); (2) fat payloads: same chains with \
a bulky `raw` dump per tool (API/file-sized), where code mode should win by returning reduced JSON; \
(3) multi-turn conversation: headline metric is final context size after 12 turns with history fed forward. \
Condense configs enable summarize_context; standard/dynamic do not. Use example 33 (wiremock) for \
deterministic comparisons. Results vary by model.";

const EXCESS_WORK_FOOTNOTE: &str =
    "* excess tool calls vs expected chain; token comparison not like-for-like";

fn flag_excess_work(label: String, tool_calls: u32, expected_tool_count: usize) -> String {
    if tool_calls as usize > expected_tool_count && label != "—" && label != "ERR" {
        format!("{label}*")
    } else {
        label
    }
}

pub fn print_benchmark_table(rows: &[BenchmarkRow], baseline_standard_total: u32) {
    println!(
        "| {:<32} | {:>10} | {:>10} | {:>9} | {:>10} | {:>11} | {:>11} | {:>12} | {:>4} |",
        "config",
        "llm_rounds",
        "tool_calls",
        "peak_msgs",
        "peak_tools",
        "cum_total",
        "cum_prompt",
        "vs_standard",
        "done",
    );
    println!(
        "|{}|{}|{}|{}|{}|{}|{}|{}|{}|",
        "-".repeat(34),
        "-".repeat(12),
        "-".repeat(12),
        "-".repeat(11),
        "-".repeat(12),
        "-".repeat(13),
        "-".repeat(13),
        "-".repeat(14),
        "-".repeat(6),
    );

    let mut saw_excess = false;

    for row in rows {
        if let Some(err) = &row.error {
            println!(
                "| {:<32} | {:>10} | {:>10} | {:>9} | {:>10} | {:>11} | {:>11} | {:>12} | {:>4} |",
                row.label, err, 0, 0, 0, 0, 0, "ERR", "—",
            );
            continue;
        }

        let kind = config_for_label(&row.label)
            .map(BenchConfig::kind)
            .unwrap_or(ConfigKind::Standard);

        let vs_standard =
            bench_configs::token_savings_label(kind, baseline_standard_total, row.cum_total_tokens);

        let expected_tools = profile_for_label(&row.label)
            .map(ConversationProfile::expected_tools)
            .map(|t| t.len())
            .unwrap_or(0);

        let vs_standard = if kind != ConfigKind::Standard {
            let flagged = flag_excess_work(vs_standard, row.tool_calls, expected_tools);
            if flagged.ends_with('*') {
                saw_excess = true;
            }
            flagged
        } else {
            vs_standard
        };

        let done = if row.completed { "Y" } else { "N" };

        println!(
            "| {:<32} | {:>10} | {:>10} | {:>9} | {:>10} | {:>11} | {:>11} | {:>12} | {:>4} |",
            row.label,
            row.llm_rounds,
            row.tool_calls,
            row.peak_context_msgs,
            row.peak_tools,
            row.cum_total_tokens,
            row.cum_prompt_tokens,
            vs_standard,
            done,
        );
    }

    if saw_excess {
        println!("\n{EXCESS_WORK_FOOTNOTE}");
    }
}

/// Data for one multi-turn context-growth row (decoupled from wiremock driver).
#[derive(Debug, Clone)]
pub struct MultiTurnReport {
    pub final_context_msgs: usize,
    pub final_context_tokens: u32,
    pub peak_context_tokens: u32,
    pub cum_total_tokens: u32,
    pub per_turn_context_msgs: Vec<usize>,
    pub turn_errors: u32,
}

pub fn print_multiturn_table(rows: &[(String, MultiTurnReport)]) {
    let baseline_final = rows
        .iter()
        .find(|(label, _)| label.ends_with("_standard"))
        .map(|(_, m)| m.final_context_tokens)
        .unwrap_or(0);

    println!(
        "| {:<32} | {:>9} | {:>11} | {:>10} | {:>11} | {:>12} | {:>6} | {}",
        "config",
        "final_msgs",
        "final_tokens",
        "peak_tokens",
        "cum_total",
        "vs_standard",
        "errors",
        "ctx_sizes",
    );
    println!(
        "|{}|{}|{}|{}|{}|{}|{}|{}|",
        "-".repeat(34),
        "-".repeat(11),
        "-".repeat(13),
        "-".repeat(12),
        "-".repeat(13),
        "-".repeat(14),
        "-".repeat(8),
        "-".repeat(20),
    );

    for (label, m) in rows {
        let kind = config_for_label(label)
            .map(BenchConfig::kind)
            .unwrap_or(ConfigKind::Standard);
        let vs_standard = if kind == ConfigKind::Standard {
            "—".to_string()
        } else {
            bench_configs::token_savings_label(kind, baseline_final, m.final_context_tokens)
        };
        let progression = format!(
            "[{}]",
            m.per_turn_context_msgs
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "| {:<32} | {:>9} | {:>11} | {:>10} | {:>11} | {:>12} | {:>6} | {}",
            label,
            m.final_context_msgs,
            m.final_context_tokens,
            m.peak_context_tokens,
            m.cum_total_tokens,
            vs_standard,
            m.turn_errors,
            progression,
        );
    }
}
