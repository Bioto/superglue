//! Example 10: Rich structured JSON from multiple tools.

mod support;

use serde_json::json;
use support::{model, require_api_key, FnTool, usage_line};
use superglue::tools::ToolSpec;

fn quote(ticker: &str) -> Option<serde_json::Value> {
    match ticker {
        "AAPL" => Some(json!({ "price": 227.52, "change": 1.34, "change_pct": 0.59, "volume": 48210300 })),
        "MSFT" => Some(json!({ "price": 415.8, "change": -2.1, "change_pct": -0.5, "volume": 22100000 })),
        "NVDA" => Some(json!({ "price": 875.4, "change": 12.05, "change_pct": 1.4, "volume": 61500000 })),
        _ => None,
    }
}

fn company_info(ticker: &str) -> Option<serde_json::Value> {
    match ticker {
        "AAPL" => Some(json!({ "name": "Apple Inc.", "sector": "Technology", "employees": 161000, "hq": "Cupertino, CA" })),
        "MSFT" => Some(json!({ "name": "Microsoft Corporation", "sector": "Technology", "employees": 221000, "hq": "Redmond, WA" })),
        "NVDA" => Some(json!({ "name": "NVIDIA Corporation", "sector": "Semiconductors", "employees": 29600, "hq": "Santa Clara, CA" })),
        _ => None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .system_prompt(
            "You are a financial assistant. Always use the available tools to retrieve real data before answering.",
        )
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "get_stock_quote".into(),
                    description: Some("Retrieve the latest stock quote for a ticker symbol.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": { "ticker": { "type": "string" } },
                        "required": ["ticker"]
                    }),
                    static_tool: false,
                },
                |args| {
                    let t = args
                        .get("ticker")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_uppercase();
                    match quote(&t) {
                        Some(q) => Ok(json!({ "ticker": t, "currency": "USD", "quote": q })),
                        None => Ok(json!({ "error": format!("Ticker '{t}' not found") })),
                    }
                },
            )
            .into_arc(),
        )
        .await?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "get_company_info".into(),
                    description: Some("Get basic information about a publicly listed company.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": { "ticker": { "type": "string" } },
                        "required": ["ticker"]
                    }),
                    static_tool: false,
                },
                |args| {
                    let t = args
                        .get("ticker")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_uppercase();
                    match company_info(&t) {
                        Some(info) => Ok(json!({ "ticker": t, "info": info })),
                        None => Ok(json!({ "error": format!("Ticker '{t}' not found") })),
                    }
                },
            )
            .into_arc(),
        )
        .await?;

    let outcome = client
        .complete(
            "Give me a brief summary of Apple (AAPL) — current price, today's change, and a couple of key company facts.",
            superglue::CallOptions::default(),
        )
        .await?;

    println!("{}", outcome.content.as_deref().unwrap_or(""));
    println!("rounds: {}", outcome.rounds);
    println!("{}", usage_line(&outcome));
    Ok(())
}
