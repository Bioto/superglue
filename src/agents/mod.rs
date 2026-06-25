//! Agent system — define agents with personas, goals, and constraints; run them
//! against user messages with the full superglue tool loop, hooks, and guardrails.
//!
//! # Quick-start
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use superglue::agents::{AgentEngine, AgentSpec};
//! use superglue::chat::ChatOptions;
//! use superglue::guardrails::GuardrailRegistry;
//! use superglue::hooks::HookRegistry;
//! use superglue::http::{ClientConfig, HttpClient};
//! use superglue::tools::ToolRegistry;
//!
//! async fn example() {
//!     let spec = AgentSpec::new("Research Assistant", "You are an expert research assistant.")
//!         .with_goal("Provide accurate, well-sourced answers.")
//!         .with_constraint("Never speculate without clearly labelling it as such.");
//!
//!     let http = HttpClient::new(ClientConfig::default()).unwrap();
//!     let tools = ToolRegistry::new();
//!     let options = ChatOptions::new("https://api.openai.com", "sk-...", "gpt-5.4-nano-2026-03-17-mini");
//!
//!     let engine = AgentEngine::new(spec);
//!     let outcome = engine.run(&http, &tools, "What is Rust's ownership model?", &options).await.unwrap();
//!     println!("{}", outcome.content.unwrap_or_default());
//! }
//! ```

use std::fmt::Write as FmtWrite;
use std::sync::Arc;

use tracing::instrument;

use crate::chat::{
    ChatError, ChatOptions, CompletionOutcome, StreamOutcome, complete_with_tools, stream_complete,
};
use crate::guardrails::GuardrailRegistry;
use crate::hooks::HookRegistry;
use crate::http::HttpClient;
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// AgentSpec
// ---------------------------------------------------------------------------

/// Declarative configuration for an agent.
///
/// An `AgentSpec` captures who the agent is (persona), what it aims to do
/// (goals), detailed workflow instructions (e.g. investigation steps, output
/// format), and what it must not do (constraints). Calling
/// [`AgentSpec::compile_system_prompt`] converts these into a structured
/// system prompt that is prepended to every LLM call.
///
/// If you want full control over the system prompt text, set
/// [`AgentSpec::system_prompt_override`] instead — the persona/goals/
/// instructions/constraints fields are then ignored during compilation.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    /// Unique identifier for this agent (used in logs and traces).
    pub name: String,
    /// Personality and role description. Becomes the opening sentence of the
    /// compiled system prompt: "You are {persona}." Do not prefix with "You are".
    pub persona: String,
    /// High-level objectives the agent should pursue.
    pub goals: Vec<String>,
    /// Detailed workflow instructions (e.g. investigation steps, output format).
    pub instructions: Vec<String>,
    /// Hard rules the agent must not violate.
    pub constraints: Vec<String>,
    /// LLM model to use (e.g. `"gpt-5.4-nano-2026-03-17-mini"`).
    /// Overrides the model in [`ChatOptions`] when non-empty.
    pub model: String,
    /// Maximum number of LLM + tool-call rounds per `run()`. Default 16.
    pub max_tool_rounds: u32,
    /// Maximum output-guardrail retry attempts per `run()`. Default 3.
    pub max_output_retries: u32,
    /// When `Some`, the compiled system prompt is replaced entirely with this
    /// value. Useful for agents whose prompt is managed externally.
    pub system_prompt_override: Option<String>,
    /// Optional reasoning effort override for reasoning-capable models.
    pub reasoning_effort: Option<String>,
}

impl AgentSpec {
    /// Create a minimal agent spec with just a name and persona.
    pub fn new(name: impl Into<String>, persona: impl Into<String>) -> Self {
        AgentSpec {
            name: name.into(),
            persona: persona.into(),
            goals: Vec::new(),
            instructions: Vec::new(),
            constraints: Vec::new(),
            model: String::new(),
            max_tool_rounds: 16,
            max_output_retries: 3,
            system_prompt_override: None,
            reasoning_effort: None,
        }
    }

    /// Append a goal (chainable).
    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goals.push(goal.into());
        self
    }

    /// Append a workflow instruction (chainable).
    ///
    /// Instruction blocks are rendered as prose under an `## Instructions`
    /// section, suitable for multi-step workflows and output templates.
    pub fn with_instructions(mut self, instruction: impl Into<String>) -> Self {
        self.instructions.push(instruction.into());
        self
    }

    /// Append a constraint (chainable).
    pub fn with_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Override the model for this agent (chainable).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set max tool rounds (chainable).
    pub fn with_max_tool_rounds(mut self, n: u32) -> Self {
        self.max_tool_rounds = n;
        self
    }

    /// Replace the compiled system prompt with a verbatim string (chainable).
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt_override = Some(prompt.into());
        self
    }

    /// Compile the agent's persona, goals, instructions, and constraints into a
    /// structured system prompt string.
    ///
    /// If [`system_prompt_override`](AgentSpec::system_prompt_override) is set,
    /// that value is returned unchanged.
    ///
    /// # Output format
    ///
    /// ```text
    /// You are {persona}.
    ///
    /// ## Goals
    /// - {goal_1}
    /// - {goal_2}
    ///
    /// ## Instructions
    /// {instruction_block_1}
    ///
    /// ## Constraints
    /// - {constraint_1}
    /// ```
    pub fn compile_system_prompt(&self) -> String {
        if let Some(ov) = &self.system_prompt_override {
            return ov.clone();
        }

        let mut out = format!("You are {}.", self.persona.trim_end_matches('.'));

        if !self.goals.is_empty() {
            out.push_str("\n\n## Goals\n");
            for g in &self.goals {
                let _ = writeln!(out, "- {g}");
            }
        }

        if !self.instructions.is_empty() {
            out.push_str("\n\n## Instructions\n\n");
            for i in &self.instructions {
                out.push_str(i.trim());
                out.push_str("\n\n");
            }
        }

        if !self.constraints.is_empty() {
            out.push_str("## Constraints\n");
            for c in &self.constraints {
                let _ = writeln!(out, "- {c}");
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// AgentEngine
// ---------------------------------------------------------------------------

/// Execution primitive for a superglue agent.
///
/// `AgentEngine` wraps an [`AgentSpec`] with optional [`HookRegistry`] and
/// [`GuardrailRegistry`], and exposes `run` / `stream` methods that call
/// the underlying [`complete_with_tools`] / [`stream_complete`] functions
/// with a pre-compiled system prompt injected into [`ChatOptions`].
pub struct AgentEngine {
    /// The agent's configuration.
    pub spec: AgentSpec,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
}

impl AgentEngine {
    /// Create an engine from a spec, using empty hook and guardrail registries.
    pub fn new(spec: AgentSpec) -> Self {
        AgentEngine {
            spec,
            hooks: Arc::new(HookRegistry::new()),
            guardrails: Arc::new(GuardrailRegistry::new()),
        }
    }

    /// Attach a hook registry (builder-style).
    pub fn with_hooks(mut self, hooks: Arc<HookRegistry>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Attach a guardrail registry (builder-style).
    pub fn with_guardrails(mut self, guardrails: Arc<GuardrailRegistry>) -> Self {
        self.guardrails = guardrails;
        self
    }

    /// Build the effective [`ChatOptions`] from `base_options`, injecting the
    /// compiled system prompt and the agent's model (if non-empty).
    fn effective_options(&self, base_options: &ChatOptions) -> ChatOptions {
        let mut opts = base_options.clone();
        opts.system_prompt = Some(self.spec.compile_system_prompt());
        if !self.spec.model.is_empty() {
            opts.model = self.spec.model.clone();
        }
        if let Some(re) = &self.spec.reasoning_effort {
            opts.reasoning_effort = Some(re.clone());
        }
        opts.max_tool_rounds = self.spec.max_tool_rounds;
        opts
    }

    /// Run a single-turn completion (with tool loop).
    ///
    /// The agent's compiled system prompt is prepended to `base_options`; all
    /// other fields (api_key, base_url, temperature, …) are taken from
    /// `base_options` as-is. The agent's `model` overrides `base_options.model`
    /// when set.
    #[instrument(
        skip(self, http, registry, base_options, user_message),
        fields(
            agent.name = %self.spec.name,
            agent.model = %self.spec.model,
            agent.max_tool_rounds = self.spec.max_tool_rounds
        )
    )]
    pub async fn run(
        &self,
        http: &HttpClient,
        registry: &ToolRegistry,
        user_message: impl Into<String>,
        base_options: &ChatOptions,
    ) -> Result<CompletionOutcome, ChatError> {
        use crate::openai::ChatMessage;
        let opts = self.effective_options(base_options);
        let messages = vec![ChatMessage::text("user", user_message.into())];
        complete_with_tools(
            http,
            registry,
            &self.hooks,
            &self.guardrails,
            messages,
            &opts,
        )
        .await
    }

    /// Run a streaming completion (no tool loop).
    ///
    /// Each content token is delivered to `on_delta` as it arrives.
    #[instrument(
        skip(self, http, base_options, on_delta, user_message),
        fields(
            agent.name = %self.spec.name,
            agent.model = %self.spec.model
        )
    )]
    pub async fn stream<F>(
        &self,
        http: &HttpClient,
        user_message: impl Into<String>,
        base_options: &ChatOptions,
        on_delta: F,
    ) -> Result<StreamOutcome, ChatError>
    where
        F: FnMut(String) + Send,
    {
        use crate::openai::ChatMessage;
        let opts = self.effective_options(base_options);
        let messages = vec![ChatMessage::text("user", user_message.into())];
        stream_complete(
            http,
            &self.hooks,
            &self.guardrails,
            messages,
            &opts,
            on_delta,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Unit tests (no HTTP)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_prompt_persona_only() {
        let spec = AgentSpec::new("bot", "a helpful assistant");
        let prompt = spec.compile_system_prompt();
        assert!(prompt.contains("You are a helpful assistant."), "{prompt}");
        assert!(!prompt.contains("## Goals"), "no goals section expected");
    }

    #[test]
    fn compile_prompt_with_goals_and_constraints() {
        let spec = AgentSpec::new("bot", "a researcher")
            .with_goal("Provide accurate answers")
            .with_goal("Cite sources")
            .with_constraint("Never speculate");
        let prompt = spec.compile_system_prompt();
        assert!(prompt.contains("## Goals"), "{prompt}");
        assert!(prompt.contains("- Provide accurate answers"), "{prompt}");
        assert!(prompt.contains("- Cite sources"), "{prompt}");
        assert!(prompt.contains("## Constraints"), "{prompt}");
        assert!(prompt.contains("- Never speculate"), "{prompt}");
    }

    #[test]
    fn compile_prompt_with_instructions() {
        let spec = AgentSpec::new("bot", "a planner")
            .with_goal("Deliver actionable plans")
            .with_instructions("Step 1: investigate.\nStep 2: write the plan.")
            .with_constraint("Do not implement");
        let prompt = spec.compile_system_prompt();
        assert!(prompt.contains("## Instructions"), "{prompt}");
        assert!(prompt.contains("Step 1: investigate."), "{prompt}");
        assert!(prompt.contains("Step 2: write the plan."), "{prompt}");
        let goals_pos = prompt.find("## Goals").unwrap();
        let instructions_pos = prompt.find("## Instructions").unwrap();
        let constraints_pos = prompt.find("## Constraints").unwrap();
        assert!(goals_pos < instructions_pos);
        assert!(instructions_pos < constraints_pos);
    }

    #[test]
    fn compile_prompt_override_bypasses_compilation() {
        let spec = AgentSpec::new("bot", "irrelevant persona")
            .with_goal("ignored goal")
            .with_system_prompt("Custom verbatim prompt.");
        let prompt = spec.compile_system_prompt();
        assert_eq!(prompt, "Custom verbatim prompt.");
    }

    #[test]
    fn builder_setters_are_chainable() {
        let spec = AgentSpec::new("bot", "a coder")
            .with_model("gpt-5.4-nano-2026-03-17")
            .with_max_tool_rounds(8)
            .with_constraint("Write idiomatic Rust");
        assert_eq!(spec.model, "gpt-5.4-nano-2026-03-17");
        assert_eq!(spec.max_tool_rounds, 8);
        assert_eq!(spec.constraints.len(), 1);
    }
}
