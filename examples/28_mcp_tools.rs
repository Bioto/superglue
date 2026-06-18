//! Example 28: MCP tools alongside local Rust tools (stdio MCP server).

mod support;

use support::client_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("MCP_RUN").is_err() {
        println!("Skip MCP example (set MCP_RUN=1 and ensure npx is available)");
        return Ok(());
    }

    let client = client_from_env()?;
    client
        .connect_mcp_stdio(
            "npx",
            vec![
                "-y".into(),
                "@modelcontextprotocol/server-everything".into(),
            ],
            None,
            Some("mcp".into()),
        )
        .await?;

    let outcome = client
        .complete(
            "List one MCP tool name you have available (prefix mcp__). Reply in one line.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("content: {:?}", outcome.content);
    Ok(())
}
