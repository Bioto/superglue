//! Optional live demo: one chat completion with tools against an OpenAI-compatible API.
//!
//! Run (requires network + API key):
//!
//! ```text
//! export OPENAI_API_KEY=sk-...
//! # optional:
//! export OPENAI_BASE_URL=https://api.openai.com
//! export OPENAI_MODEL=gpt-4o-mini
//! cargo run --example openai_chat
//! ```
//!
//! If `OPENAI_API_KEY` is unset, prints instructions and exits successfully.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{Tool, ToolRegistry, ToolSpec};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: Some("Echo back the provided message.".to_string()),
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Text to echo back" }
                },
                "required": ["message"]
            }),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = match std::env::var("OPENAI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!(
                "Set OPENAI_API_KEY to run this example. Optional: OPENAI_BASE_URL, OPENAI_MODEL."
            );
            return Ok(());
        }
    };

    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let http = HttpClient::new(ClientConfig::default())?;
    let registry = ToolRegistry::new();
    registry.register(Arc::new(EchoTool)).await?;

    let opts = ChatOptions {
        base_url,
        api_key: secrecy::Secret::new(key),
        model,
        max_tool_rounds: 8,
        system_prompt: Some(
            "You are a concise assistant that echoes whatever the user asks you to echo."
                .to_string(),
        ),
        temperature: Some(0.2),
        ..Default::default()
    };

    let messages = vec![ChatMessage::text(
        "user",
        "Call the echo tool once with message \"hello from superglue example\". \
         After you get the tool result, reply with one short sentence.",
    )];

    let outcome = complete_with_tools(
        &http,
        &registry,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        messages,
        &opts,
    )
    .await?;
    println!("rounds:  {}", outcome.rounds);
    println!("content: {:?}", outcome.content);
    println!("finish:  {:?}", outcome.finish_reason);
    if let Some(usage) = outcome.usage {
        println!(
            "usage — prompt: {}, completion: {}, total: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }
    Ok(())
}
