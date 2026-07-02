//! Example 33: Context optimization benchmark — compares configs via wiremock (no API key).

#[path = "../benches/support/context_fixtures.rs"]
mod context_fixtures;

use context_fixtures::{
    chat_options_for_run, print_comparison_table, print_multiturn_comparison,
    run_multiturn_wiremock, run_scenario, BENCH_CONFIGS, ConversationProfile, ScenarioMetrics,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Context optimization benchmark (wiremock, GlueLLM-style tool chains)\n");

    for profile in ConversationProfile::ALL {
        println!("=== {} scenario ===\n", profile.name());
        let mut rows: Vec<(String, ScenarioMetrics)> = Vec::new();

        for cfg in BENCH_CONFIGS {
            let opts = chat_options_for_run(cfg, *profile);
            let (_outcome, metrics) = run_scenario(opts, *profile).await?;
            rows.push((format!("{}_{}", profile.name(), cfg.label), metrics));
        }

        print_comparison_table(&rows);
        println!();
    }

    println!("=== multi-turn conversation (context growth) ===\n");
    let mut multiturn_rows = Vec::new();
    for cfg in BENCH_CONFIGS {
        let metrics = run_multiturn_wiremock(cfg).await?;
        multiturn_rows.push((format!("multiturn_{}", cfg.label), metrics));
    }
    print_multiturn_comparison(&multiturn_rows);
    println!();

    Ok(())
}
