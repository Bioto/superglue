//! Example 19: Guardrails (blocklist, max length, PII, custom handlers).

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use superglue::guardrails::{
    BlocklistAction, GuardrailHandler, GuardrailOutcome, GuardrailRegistry, GuardrailStage,
    LengthStrategy,
};
use support::{model, require_api_key};

struct CompetitorGuardrail;

#[async_trait]
impl GuardrailHandler for CompetitorGuardrail {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let competitors = ["competitor_a", "rival_corp", "other_llm"];
        let lower = content.to_lowercase();
        for c in competitors {
            if lower.contains(c) {
                return GuardrailOutcome::Block(format!("competitor mention detected: '{c}'"));
            }
        }
        GuardrailOutcome::Allow(content.to_string())
    }
}

struct ApiKeyRedactGuardrail;

#[async_trait]
impl GuardrailHandler for ApiKeyRedactGuardrail {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let re = regex::Regex::new(r"\bsk-[a-zA-Z0-9]{20,}\b").unwrap();
        GuardrailOutcome::Allow(re.replace_all(content, "[API_KEY_REDACTED]").into_owned())
    }
}

struct TokenRedactGuardrail;

#[async_trait]
impl GuardrailHandler for TokenRedactGuardrail {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let re = regex::Regex::new(r"token=[A-Za-z0-9]+").unwrap();
        GuardrailOutcome::Allow(re.replace_all(content, "token=[REDACTED]").into_owned())
    }
}

struct QualityChecker {
    attempts: std::sync::atomic::AtomicU32,
}

#[async_trait]
impl GuardrailHandler for QualityChecker {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        let n = self
            .attempts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if content.len() < 30 && n < 2 {
            GuardrailOutcome::Block(format!(
                "response too short ({} chars), please elaborate",
                content.len()
            ))
        } else {
            GuardrailOutcome::Allow(content.to_string())
        }
    }
}

async fn section<F: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>>(
    title: &str,
    f: F,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n──── {title} ────");
    f.await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = require_api_key();
    let model = model();
    let both = [GuardrailStage::Input, GuardrailStage::Output];
    let input_only = [GuardrailStage::Input];
    let output_only = [GuardrailStage::Output];

    section("1. Blocklist guardrail (input block)", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client
            .add_blocklist_guardrail(
                &["profanity".to_string(), r"\bbadword\b".to_string()],
                BlocklistAction::Block,
                &input_only,
                "profanity-filter",
            )
            .await?;
        match client
            .complete(
                "Please say the word badword.",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content: {:?}", o.content),
            Err(e) => println!("Blocked as expected: {e}"),
        }
        Ok(())
    })
    .await?;

    section("2. Blocklist guardrail (output redact)", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client
            .add_blocklist_guardrail(
                &[r"\b\d{3}-\d{2}-\d{4}\b".to_string()],
                BlocklistAction::Redact,
                &output_only,
                "ssn-redact",
            )
            .await?;
        match client
            .complete(
                "Pretend your SSN is 123-45-6789 and mention it.",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content (redacted): {:?}", o.content),
            Err(e) => println!("Error: {e}"),
        }
        Ok(())
    })
    .await?;

    section("3. Max-length guardrail", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client
            .add_max_length_guardrail(
                Some(50),
                Some(200),
                LengthStrategy::Truncate,
                "length-limits",
            )
            .await;
        let long = "A".repeat(100);
        match client
            .complete(long, superglue::CallOptions::default())
            .await
        {
            Ok(o) => {
                let c = o.content.unwrap_or_default();
                println!(
                    "Content (truncated): {}...",
                    c.chars().take(80).collect::<String>()
                );
            }
            Err(e) => println!("Blocked long input: {e}"),
        }
        Ok(())
    })
    .await?;

    section("4. PII redaction guardrail", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client.add_pii_guardrail(&output_only, "pii-filter").await;
        match client
            .complete(
                "Summarize: call me at 555-123-4567 or email me@example.com",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content (PII redacted): {:?}", o.content),
            Err(e) => println!("Error: {e}"),
        }
        Ok(())
    })
    .await?;

    section("5. Custom guardrail function", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client
            .register_guardrail(
                &input_only,
                "competitor-filter",
                Arc::new(CompetitorGuardrail),
            )
            .await;
        match client
            .complete(
                "Tell me about competitor_a.",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content: {:?}", o.content),
            Err(e) => println!("Blocked by custom guardrail: {e}"),
        }
        match client
            .complete(
                "Tell me about large language models in general.",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!(
                "Safe message — content: {}",
                o.content
                    .unwrap_or_default()
                    .chars()
                    .take(80)
                    .collect::<String>()
            ),
            Err(e) => println!("Unexpected error: {e}"),
        }
        Ok(())
    })
    .await?;

    section("6. Custom output transform", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        client
            .register_guardrail(
                &output_only,
                "api-key-redact",
                Arc::new(ApiKeyRedactGuardrail),
            )
            .await;
        match client
            .complete(
                "Please echo the text: sk-abcdefghijklmnopqrstuvwxyz",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content (API key redacted): {:?}", o.content),
            Err(e) => println!("Error: {e}"),
        }
        Ok(())
    })
    .await?;

    section("7. Output guardrail with retry loop", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .max_output_retries(3)
            .build()?;
        let checker = QualityChecker {
            attempts: std::sync::atomic::AtomicU32::new(0),
        };
        client
            .register_guardrail(&output_only, "quality-checker", Arc::new(checker))
            .await;
        match client
            .complete(
                "Say 'hi' in exactly two letters.",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!("Content after retry: {:?}", o.content),
            Err(e) => println!("All retries exhausted: {e}"),
        }
        Ok(())
    })
    .await?;

    section("8. Guardrails and hooks together", async {
        let client = superglue::Client::builder()
            .api_key(api_key.clone())
            .model(model.clone())
            .build()?;
        use superglue::hooks::{HookConfig, HookErrorStrategy, HookHandler, HookStage};
        struct PreLog;
        #[async_trait]
        impl HookHandler for PreLog {
            async fn execute(
                &self,
                ctx: superglue::hooks::HookContext,
            ) -> Result<superglue::hooks::HookContext, superglue::hooks::HookError> {
                println!(
                    "  [hook] PRE_COMPLETION — prompt={}",
                    ctx.content.chars().take(60).collect::<String>()
                );
                Ok(ctx)
            }
        }
        client
            .register_hook(
                HookStage::PreCompletion,
                HookConfig {
                    name: "logger".into(),
                    error_strategy: HookErrorStrategy::Skip,
                    handler: Arc::new(PreLog),
                },
            )
            .await;
        client
            .register_guardrail(&input_only, "token-redact", Arc::new(TokenRedactGuardrail))
            .await;
        match client
            .complete(
                "My token=supersecret123, what should I do with it?",
                superglue::CallOptions::default(),
            )
            .await
        {
            Ok(o) => println!(
                "  Content: {}",
                o.content
                    .unwrap_or_default()
                    .chars()
                    .take(80)
                    .collect::<String>()
            ),
            Err(e) => println!("  Error: {e}"),
        }
        Ok(())
    })
    .await?;

    let _ = GuardrailRegistry::new();
    let _ = both;

    println!("\nAll guardrail examples completed.");
    Ok(())
}
