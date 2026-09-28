//! Ordering invariants for context-optimization benchmarks (wiremock).

#[path = "../benches/support/context_fixtures.rs"]
mod context_fixtures;

use context_fixtures::{
    BENCH_CONFIGS, ConversationProfile, FAT_RAW_CHARS, chat_options_for_fat_run,
    run_scenario_with_raw,
};

#[tokio::test]
async fn fat_payloads_code_mode_uses_fewer_tokens_than_standard() {
    let profile = ConversationProfile::LongChain;
    let standard_cfg = BENCH_CONFIGS
        .iter()
        .find(|cfg| cfg.label == "standard")
        .expect("standard config");
    let code_cfg = BENCH_CONFIGS
        .iter()
        .find(|cfg| cfg.label == "code")
        .expect("code config");
    let code_condense_cfg = BENCH_CONFIGS
        .iter()
        .find(|cfg| cfg.label == "code_condense")
        .expect("code_condense config");

    let (_std_out, standard) = run_scenario_with_raw(
        chat_options_for_fat_run(standard_cfg, profile),
        profile,
        FAT_RAW_CHARS,
    )
    .await
    .expect("standard fat scenario");
    let (_code_out, code) = run_scenario_with_raw(
        chat_options_for_fat_run(code_cfg, profile),
        profile,
        FAT_RAW_CHARS,
    )
    .await
    .expect("code fat scenario");
    let (_cc_out, code_condense) = run_scenario_with_raw(
        chat_options_for_fat_run(code_condense_cfg, profile),
        profile,
        FAT_RAW_CHARS,
    )
    .await
    .expect("code_condense fat scenario");

    assert!(standard.completed, "standard fat chain should complete");
    assert!(code.completed, "code fat chain should complete");
    assert!(
        code_condense.completed,
        "code_condense fat chain should complete"
    );

    assert!(
        code.cum_total_tokens < standard.cum_total_tokens,
        "code ({}) should beat standard ({}) on fat payloads",
        code.cum_total_tokens,
        standard.cum_total_tokens
    );
    assert!(
        code_condense.cum_total_tokens < standard.cum_total_tokens,
        "code_condense ({}) should beat standard ({}) on fat payloads",
        code_condense.cum_total_tokens,
        standard.cum_total_tokens
    );
    assert!(
        code.history_bytes < standard.history_bytes,
        "code history ({} bytes) should be smaller than standard ({} bytes)",
        code.history_bytes,
        standard.history_bytes
    );
    let blob = "x".repeat(256);
    let standard_has_dump = serde_json::to_string(&_std_out.messages)
        .unwrap_or_default()
        .contains(&blob);
    let code_has_dump = serde_json::to_string(&_code_out.messages)
        .unwrap_or_default()
        .contains(&blob);
    assert!(
        standard_has_dump,
        "standard chat should retain the raw dump"
    );
    assert!(
        !code_has_dump,
        "code chat should not re-enter the raw dump after JS reduction"
    );
}
