//! Example 22: Process events via [`StatusEmitter`].

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use superglue::events::{ProcessEvent, StatusEmitter, StatusSubscriber};
use support::{model, require_api_key};

struct PrintSubscriber;

#[async_trait]
impl StatusSubscriber for PrintSubscriber {
    async fn on_event(&self, event: ProcessEvent) {
        let cost = event
            .estimated_cost_usd
            .map(|c| format!("${c:.6}"))
            .unwrap_or_else(|| "n/a".to_string());
        let extra = if let Some(err) = &event.error_type {
            format!(" error={err:?}")
        } else if event.tool_call_count > 0 {
            format!(" tools={}", event.tool_call_count)
        } else {
            String::new()
        };
        println!(
            "[{}] model={} round={} cost={}{}",
            event.kind.as_str(),
            event.model,
            event.round,
            cost,
            extra
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let emitter = Arc::new(StatusEmitter::new());
    emitter
        .subscribe(Arc::new(PrintSubscriber) as Arc<dyn StatusSubscriber>)
        .await;

    let client = superglue::Client::builder()
        .api_key(require_api_key())
        .model(model())
        .base_url(support::base_url())
        .max_tool_rounds(16)
        .status_emitter(emitter)
        .build()?;

    let outcome = client
        .complete("Say hello in one word.", superglue::CallOptions::default())
        .await?;

    println!("Response: {:?}", outcome.content);
    Ok(())
}
