//! Anthropic Messages API adapter.

use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::http::join_base_url;
use crate::openai::{
    ChatMessage, ContentPart, FileContent, FunctionCall, MessageContent, ToolCall,
};

use super::adapter::{
    LlmProvider, NormalizedCompletion, ProviderParseError, ProviderRequest, ProviderRequestContext,
    rate_limit_key_for,
};
use super::provider_id::ProviderId;

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone, Copy)]
pub struct AnthropicProvider;

impl AnthropicProvider {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl LlmProvider for AnthropicProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::Anthropic
    }

    fn build_chat_request(&self, ctx: &ProviderRequestContext<'_>) -> ProviderRequest {
        let base_url = ctx.credentials.base_url_for(ProviderId::Anthropic);
        let url = join_base_url(&base_url, "/v1/messages");
        let api_key = ctx
            .credentials
            .key_for(ProviderId::Anthropic)
            .expect("credentials checked before build");
        let headers = vec![
            ("x-api-key".to_string(), api_key.expose_secret().to_string()),
            (
                "anthropic-version".to_string(),
                ANTHROPIC_VERSION.to_string(),
            ),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let (system, messages) = map_messages(ctx.messages);
        let max_tokens = ctx.options.max_completion_tokens.unwrap_or(4096);

        let mut body = json!({
            "model": ctx.model_ref.model,
            "max_tokens": max_tokens,
            "messages": messages,
        });

        if let Some(sys) = system {
            body["system"] = Value::String(sys);
        } else if let Some(sp) = &ctx.options.system_prompt {
            body["system"] = Value::String(sp.clone());
        }

        if let Some(specs) = ctx.tools {
            let tools: Vec<Value> = specs
                .iter()
                .map(|s| {
                    let schema = s.parameters_schema.clone();
                    json!({
                        "name": s.name,
                        "description": s.description,
                        "input_schema": schema,
                    })
                })
                .collect();
            body["tools"] = json!(tools);
        }

        if ctx.stream {
            body["stream"] = json!(true);
        }

        if anthropic_supports_sampling_params(&ctx.model_ref.model)
            && let Some(temp) = ctx.options.temperature
        {
            body["temperature"] = json!(temp);
        }

        let rate_limit_key =
            rate_limit_key_for(ctx.model_ref, ctx.credentials).expect("credentials checked");

        ProviderRequest {
            url,
            headers,
            body,
            rate_limit_key,
            model_ref: ctx.model_ref.clone(),
            bare_model: ctx.model_ref.model.clone(),
        }
    }

    fn parse_chat_response(
        &self,
        json: &Value,
    ) -> Result<NormalizedCompletion, ProviderParseError> {
        let resp: AnthropicMessageResponse = serde_json::from_value(json.clone())?;
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        for block in &resp.content {
            match block {
                AnthropicContentBlock::Text { text } => text_parts.push(text.clone()),
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        kind: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: input.to_string(),
                        },
                    });
                }
            }
        }
        let content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };
        let usage = resp.usage.map(|u| crate::proto::Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });
        let finish_reason = if resp.stop_reason == "tool_use" {
            Some("tool_calls".to_string())
        } else {
            Some("stop".to_string())
        };
        Ok(NormalizedCompletion {
            content,
            tool_calls,
            usage,
            finish_reason,
            stop_reason: Some(resp.stop_reason),
        })
    }

    fn supports_file_upload(&self) -> bool {
        true
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageResponse {
    content: Vec<AnthropicContentBlock>,
    #[serde(default)]
    stop_reason: String,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn map_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let mut system_parts = Vec::new();
    let mut out = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(c) = &msg.content {
                    if let Some(t) = c.as_text() {
                        system_parts.push(t.to_string());
                    }
                }
            }
            "user" => {
                out.push(json!({
                    "role": "user",
                    "content": map_user_content(msg),
                }));
            }
            "assistant" => {
                let mut blocks = Vec::new();
                if let Some(c) = &msg.content {
                    if let Some(t) = c.as_text() {
                        blocks.push(json!({"type": "text", "text": t}));
                    }
                }
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        if tc.kind == "function" {
                            let input: Value =
                                serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": input,
                            }));
                        }
                    }
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            "tool" => {
                let content = msg.content.as_ref().and_then(|c| c.as_text()).unwrap_or("");
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content,
                    }],
                }));
            }
            _ => {}
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, out)
}

fn map_user_content(msg: &ChatMessage) -> Value {
    match &msg.content {
        None => json!(""),
        Some(MessageContent::Text(t)) => json!(t),
        Some(MessageContent::Parts(parts)) => {
            let blocks: Vec<Value> = parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text { text } => json!({"type": "text", "text": text}),
                    ContentPart::ImageUrl { image_url } => json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url.url,
                        }
                    }),
                    ContentPart::File { file } => map_file_part(file),
                    ContentPart::ImageRef { hash, .. } => json!({
                        "type": "text",
                        "text": format!("[unresolved image_ref: {hash}]")
                    }),
                    ContentPart::InputAudio { .. } => json!({
                        "type": "text",
                        "text": "[audio input not supported on anthropic adapter]"
                    }),
                })
                .collect();
            json!(blocks)
        }
    }
}

fn map_file_part(file: &FileContent) -> Value {
    if let Some(data) = &file.file_data {
        let media = infer_media_type(file.filename.as_deref());
        if media == "application/pdf" {
            json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media,
                    "data": data,
                }
            })
        } else {
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media,
                    "data": data,
                }
            })
        }
    } else if let Some(id) = &file.file_id {
        json!({
            "type": "text",
            "text": format!("[file_id: {id}]")
        })
    } else {
        json!({"type": "text", "text": "[empty file part]"})
    }
}

fn infer_media_type(filename: Option<&str>) -> &'static str {
    match filename {
        Some(f) if f.ends_with(".pdf") => "application/pdf",
        Some(f) if f.ends_with(".png") => "image/png",
        Some(f) if f.ends_with(".jpg") || f.ends_with(".jpeg") => "image/jpeg",
        Some(f) if f.ends_with(".gif") => "image/gif",
        Some(f) if f.ends_with(".webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

/// Newer Anthropic models reject `temperature` / `top_p` / `top_k` entirely.
fn anthropic_supports_sampling_params(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    if model.contains("sonnet-5") {
        return false;
    }
    if model.contains("opus-4-7") || model.contains("opus-4-8") {
        return false;
    }
    if model.contains("fable") {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sonnet_5_rejects_sampling_params() {
        assert!(!anthropic_supports_sampling_params("claude-sonnet-5"));
        assert!(!anthropic_supports_sampling_params(
            "anthropic:claude-sonnet-5@20260203"
        ));
    }

    #[test]
    fn older_sonnet_keeps_sampling_params() {
        assert!(anthropic_supports_sampling_params("claude-sonnet-4-6"));
        assert!(anthropic_supports_sampling_params(
            "claude-3-5-sonnet-20241022"
        ));
    }
}
