//! Example 07: `max_tool_rounds` caps the tool loop.

mod support;

use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;
use superglue::chat::ChatError;
use superglue::tools::ToolSpec;
use support::{FnTool, model, require_api_key};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    static CALL_COUNT: AtomicU32 = AtomicU32::new(0);

    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .system_prompt(
            "You are an agent that calls counter_tool repeatedly. \
             After each call check should_continue; if true, call counter_tool again immediately.",
        )
        .max_tool_rounds(3)
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "counter_tool".into(),
                    description: Some("Increment a counter and return whether to continue.".into()),
                    parameters_schema: json!({"type": "object", "properties": {}}),
                    static_tool: false,
                },
                |_| {
                    let count = CALL_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok(json!({ "count": count, "should_continue": true }))
                },
            )
            .into_arc(),
        )
        .await?;

    match client
        .complete(
            "Start counting. Keep going until you're told to stop.",
            superglue::CallOptions::default(),
        )
        .await
    {
        Ok(outcome) => {
            println!("Final content: {:?}", outcome.content);
            println!("Completion rounds: {}", outcome.rounds);
            println!("Tool calls executed: {}", CALL_COUNT.load(Ordering::SeqCst));
        }
        Err(ChatError::MaxToolRounds(n)) => {
            println!("Stopped at max_tool_rounds (expected): exceeded max tool rounds ({n})");
            println!("Tool calls executed: {}", CALL_COUNT.load(Ordering::SeqCst));
        }
        Err(e) => {
            let msg = e.to_string();
            if msg.to_lowercase().contains("max tool rounds") {
                println!("Stopped at max_tool_rounds (expected): {msg}");
                println!("Tool calls executed: {}", CALL_COUNT.load(Ordering::SeqCst));
            } else {
                return Err(e.into());
            }
        }
    }

    Ok(())
}
