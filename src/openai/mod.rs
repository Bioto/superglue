//! OpenAI-compatible chat completion types (full `POST /v1/chat/completions` spec).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::ToolSpec;

// ---------------------------------------------------------------------------
// Message content (string OR array of typed parts)
// ---------------------------------------------------------------------------

/// Image detail level for vision requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL or base64-encoded image data.
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Audio format for `input_audio` content parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioInputFormat {
    Wav,
    Mp3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    pub format: AudioInputFormat,
}

/// A single typed content part within a multipart message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    InputAudio {
        input_audio: InputAudio,
    },
    /// File input (base64 data or uploaded file id).
    File {
        file: FileContent,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Message content: either a plain string or an array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Return the text if this is a plain `Text` variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s.as_str()),
            MessageContent::Parts(_) => None,
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// "system" | "developer" | "user" | "assistant" | "tool" | "function"
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Tool calls emitted by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `role: "tool"` messages: the tool call this responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Participant name (differentiates same-role speakers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Refusal text from the model (assistant messages only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ChatMessage {
    /// Convenience constructor for a simple text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            refusal: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// `strict` mode for function tools (structured output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for parameters.
    pub parameters: Value,
    /// When `true` the model is constrained to the schema exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `tools[]` entry: `{ "type": "function", "function": { ... } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunctionDef,
}

impl From<ToolSpec> for ChatTool {
    fn from(spec: ToolSpec) -> Self {
        ChatTool {
            kind: "function".to_string(),
            function: ChatFunctionDef {
                name: spec.name,
                description: spec.description,
                parameters: spec.parameters_schema,
                strict: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tool call (in responses)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Tool choice
// ---------------------------------------------------------------------------

/// `tool_choice` value: "none" | "auto" | "required" | `{type, function}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "none" | "auto" | "required"
    Mode(String),
    /// Force a specific function: `{ "type": "function", "function": { "name": "..." } }`
    Named(NamedToolChoice),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: NamedToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

/// `response_format` options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text (default).
    Text,
    /// Any valid JSON object.
    JsonObject,
    /// JSON constrained to a specific schema.
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

/// `stop` field: one string or up to four strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>),
}

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    // Tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // Sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    // Limits
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    // Penalties
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    // Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,

    // Output format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    // Logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    // Determinism / storage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    // Service
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    // Streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    // Reasoning models (o1/o3/o4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Catch-all passthrough: any extra OpenAI parameter not explicitly typed above.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl ChatCompletionRequest {
    /// Minimal constructor used by the chat loop.
    pub fn new(model: String, messages: Vec<ChatMessage>, tools: Option<Vec<ChatTool>>) -> Self {
        ChatCompletionRequest {
            model,
            messages,
            tools,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            top_p: None,
            n: None,
            max_completion_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            response_format: None,
            logprobs: None,
            top_logprobs: None,
            seed: None,
            store: None,
            service_tier: None,
            stream: None,
            stream_options: None,
            reasoning_effort: None,
            extra: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Response — logprobs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TopLogprob {
    pub token: String,
    #[serde(default)]
    pub bytes: Option<Vec<u8>>,
    pub logprob: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChoiceLogprobs {
    #[serde(default)]
    pub content: Option<Vec<TokenLogprob>>,
    #[serde(default)]
    pub refusal: Option<Vec<TokenLogprob>>,
}

// ---------------------------------------------------------------------------
// Response — usage
// ---------------------------------------------------------------------------

/// Breakdown of completion token types.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(default)]
    pub audio_tokens: u32,
    #[serde(default)]
    pub accepted_prediction_tokens: u32,
    #[serde(default)]
    pub rejected_prediction_tokens: u32,
    #[serde(default)]
    pub cached_tokens: u32,
}

/// Usage statistics returned in the response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub completion_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    pub prompt_tokens_details: Option<TokenDetails>,
}

// ---------------------------------------------------------------------------
// Response — annotations (web search citations etc.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlCitation {
    pub start_index: u32,
    pub end_index: u32,
    pub title: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(rename = "type")]
    pub kind: String,
    pub url_citation: UrlCitation,
}

// ---------------------------------------------------------------------------
// Response — main types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    /// Unix timestamp of when the completion was created.
    #[serde(default)]
    pub created: u64,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Deprecated; backend config fingerprint.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

// ---------------------------------------------------------------------------
// Streaming response types (stream: true → Server-Sent Events)
// ---------------------------------------------------------------------------

/// One SSE payload from a `stream: true` chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Only present when `stream_options.include_usage = true` and finish_reason = "stop".
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// One choice inside a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: DeltaMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// The delta carried by each chunk — fields are absent until the model first emits them.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeltaMessage {
    #[serde(default)]
    pub role: Option<String>,
    /// Text content delta for this chunk.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool call deltas — each entry maps to a tool call by index.
    #[serde(default)]
    pub tool_calls: Option<Vec<StreamToolCallDelta>>,
    #[serde(default)]
    pub refusal: Option<String>,
}

/// A partial tool call carried by a streaming chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamToolCallDelta {
    /// Index of the tool call being accumulated (stable across chunks for the same call).
    pub index: u32,
    /// Only present in the first chunk for a given tool call.
    #[serde(default)]
    pub id: Option<String>,
    /// Only present in the first chunk for a given tool call.
    #[serde(rename = "type")]
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<StreamFunctionDelta>,
}

/// Partial function name / arguments for a streaming tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamFunctionDelta {
    /// Present only in the first chunk for this tool call.
    #[serde(default)]
    pub name: Option<String>,
    /// Argument JSON fragment — concatenate across chunks to get the full JSON.
    #[serde(default)]
    pub arguments: Option<String>,
}
