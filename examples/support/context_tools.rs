//! Register the nine-tool GlueLLM-style set used by context optimization benchmarks.

#[path = "../../benches/support/benchmark_tools.rs"]
mod benchmark_tools;

use superglue::Client;

/// Weather + forecast + flights + hotel + calc + FX + translate + country + pinned `get_time`.
pub async fn register_context_optimization_tools(
    client: &Client,
) -> Result<(), Box<dyn std::error::Error>> {
    for tool in benchmark_tools::all_benchmark_tools() {
        client.register_tool(tool).await?;
    }
    Ok(())
}
