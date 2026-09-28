//! Register the nine-tool GlueLLM-style set used by context optimization benchmarks.

#[path = "../../benches/support/benchmark_tools.rs"]
mod benchmark_tools;

use superglue::Client;

pub use benchmark_tools::FAT_RAW_CHARS;

/// Weather + forecast + flights + hotel + calc + FX + translate + country + pinned `get_time`.
pub async fn register_context_optimization_tools(
    client: &Client,
) -> Result<(), Box<dyn std::error::Error>> {
    for tool in benchmark_tools::all_benchmark_tools() {
        client.register_tool(tool).await?;
    }
    Ok(())
}

/// Same tools, each result padded with a bulky `raw` dump of `raw_chars`.
pub async fn register_fat_context_optimization_tools(
    client: &Client,
    raw_chars: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for tool in benchmark_tools::fat_benchmark_tools(raw_chars).await {
        client.register_tool(tool).await?;
    }
    Ok(())
}
