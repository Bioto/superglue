//! Regression checks for context optimization benchmark ordering invariants.

#[path = "../benches/support/context_fixtures.rs"]
mod context_fixtures;

use context_fixtures::{
    chat_options_for_run, run_multiturn_wiremock, run_scenario, BENCH_CONFIGS, ConversationProfile,
    MultiTurnMetrics,
};

async fn metrics_for(label: &str, profile: ConversationProfile) -> context_fixtures::ScenarioMetrics {
    let cfg = BENCH_CONFIGS
        .iter()
        .find(|c| c.label == label)
        .unwrap();
    let opts = chat_options_for_run(cfg, profile);
    let (_outcome, metrics) = run_scenario(opts, profile).await.unwrap();
    metrics
}

async fn multiturn_metrics_for(label: &str) -> MultiTurnMetrics {
    let cfg = BENCH_CONFIGS
        .iter()
        .find(|c| c.label == label)
        .unwrap();
    run_multiturn_wiremock(cfg).await.unwrap()
}

#[tokio::test]
async fn dynamic_condense_lowest_cum_tokens_short_chain() {
    let standard = metrics_for("standard", ConversationProfile::ShortChain).await;
    let dynamic_condense =
        metrics_for("dynamic_condense_aaak", ConversationProfile::ShortChain).await;
    assert!(
        dynamic_condense.cum_total_tokens < standard.cum_total_tokens,
        "dynamic_condense_aaak ({}) should beat standard ({}) on cum_total_tokens",
        dynamic_condense.cum_total_tokens,
        standard.cum_total_tokens
    );
}

#[tokio::test]
async fn condense_reduces_peak_context_msgs_vs_standard_short() {
    let standard = metrics_for("standard", ConversationProfile::ShortChain).await;
    let condense_plain = metrics_for("condense_plain", ConversationProfile::ShortChain).await;
    assert!(
        condense_plain.peak_context_msgs < standard.peak_context_msgs,
        "condense_plain peak_msgs ({}) should be less than standard ({})",
        condense_plain.peak_context_msgs,
        standard.peak_context_msgs
    );
}

#[tokio::test]
async fn dynamic_sends_fewer_tools_on_peak_round_than_standard_short() {
    let standard = metrics_for("standard", ConversationProfile::ShortChain).await;
    let dynamic = metrics_for("dynamic", ConversationProfile::ShortChain).await;
    assert!(
        dynamic.peak_tools < standard.peak_tools,
        "dynamic peak_tools ({}) should be less than standard ({})",
        dynamic.peak_tools,
        standard.peak_tools
    );
}

#[tokio::test]
async fn condense_aaak_beats_standard_long_chain() {
    let standard = metrics_for("standard", ConversationProfile::LongChain).await;
    let condense_aaak = metrics_for("condense_aaak", ConversationProfile::LongChain).await;
    assert!(
        condense_aaak.cum_total_tokens < standard.cum_total_tokens,
        "long condense_aaak ({}) should beat standard ({})",
        condense_aaak.cum_total_tokens,
        standard.cum_total_tokens
    );
}

#[tokio::test]
async fn dynamic_condense_beats_dynamic_long_chain() {
    let dynamic = metrics_for("dynamic", ConversationProfile::LongChain).await;
    let dynamic_condense =
        metrics_for("dynamic_condense_aaak", ConversationProfile::LongChain).await;
    assert!(
        dynamic_condense.cum_total_tokens < dynamic.cum_total_tokens,
        "long dynamic_condense_aaak ({}) should beat dynamic ({})",
        dynamic_condense.cum_total_tokens,
        dynamic.cum_total_tokens
    );
}

#[tokio::test]
async fn long_chain_all_configs_complete() {
    for cfg in BENCH_CONFIGS {
        let metrics = metrics_for(cfg.label, ConversationProfile::LongChain).await;
        assert!(
            metrics.completed,
            "{} should complete all expected tools on long chain",
            cfg.label
        );
    }
}

#[tokio::test]
async fn long_chain_uses_more_tokens_than_short_standard() {
    let short = metrics_for("standard", ConversationProfile::ShortChain).await;
    let long = metrics_for("standard", ConversationProfile::LongChain).await;
    assert!(
        long.cum_total_tokens > short.cum_total_tokens,
        "long standard ({}) should exceed short standard ({})",
        long.cum_total_tokens,
        short.cum_total_tokens
    );
}

#[tokio::test]
async fn condense_plain_beats_standard_on_cum_prompt_short() {
    let standard = metrics_for("standard", ConversationProfile::ShortChain).await;
    let condense_plain = metrics_for("condense_plain", ConversationProfile::ShortChain).await;
    assert!(
        condense_plain.cum_prompt_tokens < standard.cum_prompt_tokens,
        "condense_plain prompt tokens ({}) should beat standard ({})",
        condense_plain.cum_prompt_tokens,
        standard.cum_prompt_tokens
    );
}

#[tokio::test]
async fn dynamic_uses_more_llm_rounds_than_standard_short() {
    let standard = metrics_for("standard", ConversationProfile::ShortChain).await;
    let dynamic = metrics_for("dynamic", ConversationProfile::ShortChain).await;
    assert!(
        dynamic.llm_rounds > standard.llm_rounds,
        "dynamic rounds ({}) should exceed standard ({})",
        dynamic.llm_rounds,
        standard.llm_rounds
    );
}

#[tokio::test]
async fn condense_summarize_keeps_context_bounded() {
    let standard = multiturn_metrics_for("standard").await;
    let condense_plain = multiturn_metrics_for("condense_plain").await;

    assert!(
        condense_plain.final_context_tokens < standard.final_context_tokens,
        "condense_plain final tokens ({}) should be less than standard ({})",
        condense_plain.final_context_tokens,
        standard.final_context_tokens
    );

    let standard_grows = standard
        .per_turn_context_msgs
        .windows(2)
        .any(|w| w[1] > w[0]);
    assert!(
        standard_grows,
        "standard per-turn context should grow at least once: {:?}",
        standard.per_turn_context_msgs
    );

    assert!(
        condense_plain.final_context_tokens <= condense_plain.peak_context_tokens,
        "condense_plain final ({}) should not exceed peak ({})",
        condense_plain.final_context_tokens,
        condense_plain.peak_context_tokens
    );
}
