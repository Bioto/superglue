//! Example 18: Lifecycle hooks (observe / mutate pipeline).

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use support::{model, require_api_key, FnTool};
use superglue::hooks::{
    HookConfig, HookContext, HookError, HookErrorStrategy, HookHandler, HookRegistry, HookStage,
};
use superglue::tools::ToolSpec;

struct LogHook {
    prefix: String,
}

#[async_trait]
impl HookHandler for LogHook {
    async fn execute(&self, ctx: HookContext) -> Result<HookContext, HookError> {
        let preview = ctx.content.chars().take(60).collect::<String>();
        println!("  {}: {}", self.prefix, preview.replace('\n', " "));
        Ok(ctx)
    }
}

struct PreToolUnitHook;

#[async_trait]
impl HookHandler for PreToolUnitHook {
    async fn execute(&self, mut ctx: HookContext) -> Result<HookContext, HookError> {
        if ctx.stage == HookStage::PreTool {
            if let Ok(mut args) = serde_json::from_str::<serde_json::Value>(&ctx.content) {
                if args.get("unit").is_none() {
                    args["unit"] = json!("celsius");
                    ctx.content = args.to_string();
                }
            }
        }
        Ok(ctx)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = require_api_key();
    let model = model();

    println!("\n=== Example 1: Observation hooks (logging) ===");
    let client = superglue::Client::builder()
        .api_key(api_key.clone())
        .model(model.clone())
        .build()?;

    client
        .register_hook(
            HookStage::PreCompletion,
            HookConfig {
                name: "log-pre".into(),
                error_strategy: HookErrorStrategy::Skip,
                handler: Arc::new(LogHook {
                    prefix: "→ PRE_COMPLETION".into(),
                }),
            },
        )
        .await;

    client
        .register_hook(
            HookStage::PostCompletion,
            HookConfig {
                name: "log-post".into(),
                error_strategy: HookErrorStrategy::Skip,
                handler: Arc::new(LogHook {
                    prefix: "← POST_COMPLETION".into(),
                }),
            },
        )
        .await;

    let outcome = client
        .complete(
            "In one sentence, what is the speed of light?",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("  content: {:?}", outcome.content);

    println!("\n=== Example 2: PreTool hook — default unit ===");
    let client = superglue::Client::builder()
        .api_key(api_key.clone())
        .model(model.clone())
        .build()?;

    client
        .register_tool(
            FnTool::new(
                ToolSpec {
                    name: "get_weather".into(),
                    description: Some("Return temperature for a city.".into()),
                    parameters_schema: json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "unit": { "type": "string" }
                        },
                        "required": ["city"]
                    }),
                    static_tool: false,
                },
                |args| {
                    let city = args
                        .get("city")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let unit = args
                        .get("unit")
                        .and_then(|v| v.as_str())
                        .unwrap_or("celsius");
                    let temps = [
                        ("london", 15),
                        ("paris", 18),
                        ("new york", 22),
                        ("tokyo", 25),
                    ];
                    let temp = temps
                        .iter()
                        .find(|(c, _)| *c == city.to_lowercase())
                        .map(|(_, t)| *t)
                        .unwrap_or(20);
                    Ok(json!({ "city": city, "temperature": temp, "unit": unit }))
                },
            )
            .into_arc(),
        )
        .await?;

    client
        .register_hook(
            HookStage::PreTool,
            HookConfig {
                name: "default-unit".into(),
                error_strategy: HookErrorStrategy::Skip,
                handler: Arc::new(PreToolUnitHook),
            },
        )
        .await;

    let outcome = client
        .complete(
            "What's the temperature in Tokyo?",
            superglue::CallOptions::default(),
        )
        .await?;
    println!("  content: {:?}", outcome.content);

    println!("\n=== Example 3: HookRegistry directly ===");
    let hooks = HookRegistry::new();
    hooks
        .add(
            HookStage::PreCompletion,
            HookConfig {
                name: "direct".into(),
                error_strategy: HookErrorStrategy::Skip,
                handler: Arc::new(LogHook {
                    prefix: "registry PRE".into(),
                }),
            },
        )
        .await;

    let client = superglue::Client::builder()
        .api_key(api_key)
        .model(model)
        .build()?;

    // Client already has empty hooks — demonstration only
    let _ = hooks;

    let outcome = client
        .complete("Say 'hooks ok' in two words.", superglue::CallOptions::default())
        .await?;
    println!("  content: {:?}", outcome.content);

    Ok(())
}
