//! Stateful multi-turn chat: owns caller-visible [`ChatMessage`] history.

use crate::guardrails::GuardrailRegistry;
use crate::hooks::HookRegistry;
use crate::http::HttpClient;
use crate::openai::ChatMessage;
use crate::tools::ToolRegistry;

use super::{complete_with_tools, ChatError, ChatOptions, CompletionOutcome};

/// OpenAI-style message list managed across [`Self::complete`] turns.
///
/// Does not include the synthetic `system` row from [`ChatOptions::system_prompt`];
/// that row is still prepended on each HTTP request inside [`complete_with_tools`].
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    pub messages: Vec<ChatMessage>,
}

impl Conversation {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_messages(messages: Vec<ChatMessage>) -> Self {
        Self { messages }
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.messages.push(ChatMessage::text("user", text.into()));
    }

    pub fn push_assistant_text(&mut self, text: impl Into<String>) {
        self.messages.push(ChatMessage::text("assistant", text.into()));
    }

    /// Runs the tool loop for the current [`Self::messages`] buffer and replaces it
    /// with the post-turn caller-visible history from [`CompletionOutcome::messages`].
    pub async fn complete(
        &mut self,
        http: &HttpClient,
        registry: &ToolRegistry,
        hooks: &HookRegistry,
        guardrails: &GuardrailRegistry,
        options: &ChatOptions,
    ) -> Result<CompletionOutcome, ChatError> {
        let outcome = complete_with_tools(
            http,
            registry,
            hooks,
            guardrails,
            self.messages.clone(),
            options,
        )
        .await?;
        self.messages = outcome.messages.clone();
        Ok(outcome)
    }
}
