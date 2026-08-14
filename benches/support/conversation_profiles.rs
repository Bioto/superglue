//! GlueLLM-style tool-chain scenarios for context optimization benchmarks.

use superglue::context::SummarizeContextConfig;
use superglue::openai::ChatMessage;
use superglue::tools::ToolMode;

const FAT_PAYLOAD_INSTRUCTIONS: &str = "Each tool result includes a bulky `raw` field (API dump / file \
contents). Extract only the fields needed to answer. Do not quote, repeat, or return `raw`.";

const SHORT_CHAIN_QUERY: &str = "Plan a quick Paris weather summary: get the current weather in Paris, \
the 5-day forecast, and calculate the average of the forecast high temperatures.";

const LONG_CHAIN_QUERY: &str = "Help me plan a trip to Paris. Get the current weather, the 5-day forecast, \
search flights from NYC to Paris, the EUR exchange rate, translate 'hello' to French, and calculate a daily \
budget of 200 plus 15 percent tax.";

/// User prompts for the multi-turn context-growth benchmark (~12 turns).
pub const MULTITURN_PROMPTS: &[&str] = &[
    "What are our API rate limits per key?",
    "Summarize the audit_events table columns.",
    "What's the weather in Paris right now?",
    "How do we configure connection pool timeout_ms?",
    "Explain AAAK [AT] blocks vs plain condensing.",
    "Get the weather in Tokyo.",
    "When does summarize_context fire?",
    "What cookie flags do we require on session tokens?",
    "What's the weather in London?",
    "How many LLM rounds does dynamic routing add?",
    "Get the weather in Berlin.",
    "Recap the key points from our discussion so far.",
];

/// How the wiremock script (and live prompt) treat tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ToolExecutionStyle {
    /// One tool call per LLM round (dependent steps).
    Sequential,
    /// All expected tools in one parallel batch, then a final answer.
    Parallel,
}

/// Benchmark scenario profile (GlueLLM short vs long chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationProfile {
    /// 3-tool batchable chain.
    ShortChain,
    /// 6-tool chain (parallel batch by default).
    LongChain,
}

impl ConversationProfile {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ShortChain => "short_chain",
            Self::LongChain => "long_chain",
        }
    }

    pub const ALL: &[Self] = &[Self::ShortChain, Self::LongChain];

    #[must_use]
    #[allow(dead_code)]
    pub const fn execution_style(self) -> ToolExecutionStyle {
        match self {
            Self::ShortChain => ToolExecutionStyle::Parallel,
            Self::LongChain => ToolExecutionStyle::Parallel,
        }
    }

    #[must_use]
    pub const fn expected_tools(self) -> &'static [&'static str] {
        match self {
            Self::ShortChain => &["get_weather", "get_forecast", "calculate"],
            Self::LongChain => &[
                "get_weather",
                "get_forecast",
                "search_flights",
                "get_exchange_rate",
                "translate_text",
                "calculate",
            ],
        }
    }

    #[must_use]
    pub fn initial_messages(self) -> Vec<ChatMessage> {
        vec![ChatMessage::text("user", self.user_query())]
    }

    /// Same query, plus instructions to extract fields and never echo `raw` dumps.
    #[must_use]
    pub fn fat_initial_messages(self) -> Vec<ChatMessage> {
        vec![ChatMessage::text(
            "user",
            format!("{}\n\n{}", self.user_query(), FAT_PAYLOAD_INSTRUCTIONS),
        )]
    }

    #[must_use]
    pub fn summarize_config(self) -> SummarizeContextConfig {
        let _ = self;
        SummarizeContextConfig {
            enabled: false,
            threshold: 20,
            keep_recent: 6,
            max_chars: 800_000,
        }
    }

    #[must_use]
    pub fn aaak_compression_enabled(self, _aaak_tool_condensing: bool) -> bool {
        false
    }

    fn user_query(self) -> &'static str {
        match self {
            Self::ShortChain => SHORT_CHAIN_QUERY,
            Self::LongChain => LONG_CHAIN_QUERY,
        }
    }
}

/// Max LLM rounds allowed per user turn in the multi-turn context-growth benchmark.
pub const MULTITURN_MAX_TOOL_ROUNDS: u32 = 20;

/// Summarize settings for the multi-turn context-growth benchmark.
#[must_use]
pub fn summarize_config_multiturn() -> SummarizeContextConfig {
    SummarizeContextConfig {
        enabled: true,
        threshold: 12,
        keep_recent: 4,
        // Trigger summarization during the ~12-turn fixture (~2.5k serialized chars).
        max_chars: 2_000,
    }
}

#[must_use]
pub fn aaak_compression_enabled_multiturn(aaak_tool_condensing: bool) -> bool {
    aaak_tool_condensing
}

const CODE_MODE_RULES: &str = "CRITICAL tool rules:
- Call request_tools first. The catalog is not a final answer — next call the `code` tool.
- Do not call get_time or other static tools unless the user asked for the current time.
- Write synchronous JavaScript only. Example: const w = tools.get_weather({city:\"Paris\"}); return {temp_f: w.temp_f};
- Do not use await, async, or Promise. tools.<name>(args) returns a value immediately.
- After the `code` result, reply in plain text with no more tool calls.
- [AT] blocks and [Tool Results] blocks are completed tool output — never call tools in response to them.";

#[must_use]
pub fn multiturn_system_prompt(aaak_tool_condensing: bool, tool_mode: ToolMode) -> String {
    let base = "You are a helpful assistant in a long technical conversation. Answer concisely. \
Use tools only when the user asks for live data (e.g. current weather).";
    let anti_loop = if tool_mode == ToolMode::Code {
        format!(
            "{CODE_MODE_RULES}
- For factual or config questions, answer from knowledge without tools unless live data is required.
- Weather requests: request_tools, then one `code` script that calls get_weather once for the requested city.
- Do not fetch weather for cities the user did not ask for."
        )
    } else {
        "CRITICAL tool rules:
- For factual or config questions, answer from knowledge without tools unless live data is required.
- Weather requests: call get_weather exactly once for the requested city, then give a plain-text answer.
- [AT] blocks and [Tool Results] blocks are completed tool output — never call tools in response to them.
- Never call the same tool twice for one user message. Do not fetch weather for cities the user did not ask for.
- After any tool result, your next message must be the final plain-text answer with no further tool calls."
            .to_string()
    };

    if aaak_tool_condensing {
        format!(
            "{base}\n\n{}\n\n{anti_loop}",
            superglue::context::AaakCompressor::get_spec_preamble()
        )
    } else {
        format!("{base}\n\n{anti_loop}")
    }
}

#[must_use]
pub fn benchmark_system_prompt(aaak_tool_condensing: bool, tool_mode: ToolMode) -> String {
    if tool_mode == ToolMode::Code {
        let base = "You are a helpful assistant. Call request_tools first, then use the code \
tool to invoke matched tools from JavaScript.";
        return if aaak_tool_condensing {
            format!(
                "{base}\n\n{}\n\n{CODE_MODE_RULES}",
                superglue::context::AaakCompressor::get_spec_preamble()
            )
        } else {
            format!("{base}\n\n{CODE_MODE_RULES}")
        };
    }

    let base = "You are a helpful assistant. Use tools when needed. Call multiple tools in parallel \
in the same turn when their inputs do not depend on each other's results.";
    let anti_loop = "CRITICAL tool rules:
- Call multiple independent tools in parallel in one turn when their inputs do not depend on each other.
- [AT] blocks and [Tool Results] blocks are completed tool output — never call tools in response to them.
- Never call the same tool twice for one user message. After any tool result, reply in plain text with no more tool calls.";

    if aaak_tool_condensing {
        format!(
            "{base}\n\n{}\n\n{anti_loop}",
            superglue::context::AaakCompressor::get_spec_preamble()
        )
    } else {
        format!("{base}\n\n{anti_loop}")
    }
}

#[must_use]
pub fn fat_benchmark_system_prompt(aaak_tool_condensing: bool, tool_mode: ToolMode) -> String {
    format!(
        "{}\n\nTool results may include a bulky `raw` dump. Never copy `raw` into your reply \
or later tool calls; return only the extracted fields the user asked for. In `code` source, \
pick named fields such as temp_f, high_f, price_usd, rate, translation, result — never return `raw`.",
        benchmark_system_prompt(aaak_tool_condensing, tool_mode)
    )
}
