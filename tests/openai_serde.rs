//! Serialization and deserialization tests for all OpenAI types.
//!
//! These tests verify that:
//!   - Optional fields are omitted from JSON when `None` (not sent as `null`).
//!   - All enum variants (ToolChoice, ResponseFormat, StopSequence, etc.) round-trip.
//!   - Response types parse correctly from realistic API payloads.
//!   - Streaming chunk types parse correctly.
//!   - Multipart message content serialises to the shape the API expects.

use serde_json::json;
use superglue::openai::{
    AudioInputFormat, ChatCompletionChunk, ChatCompletionRequest, ChatMessage, ChatTool,
    ContentPart, FunctionCall, ImageDetail, ImageUrl, InputAudio, JsonSchemaFormat, MessageContent,
    NamedToolChoice, NamedToolChoiceFunction, ResponseFormat, StopSequence, StreamOptions,
    ToolCall, ToolChoice,
};

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

#[test]
fn chat_message_text_omits_none_fields() {
    let msg = ChatMessage::text("user", "hello");
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["role"], "user");
    assert_eq!(v["content"], "hello");
    // Optional fields must be absent, not null
    assert!(v.get("tool_calls").is_none());
    assert!(v.get("tool_call_id").is_none());
    assert!(v.get("name").is_none());
    assert!(v.get("refusal").is_none());
}

#[test]
fn chat_message_system_role() {
    let msg = ChatMessage::text("system", "You are helpful.");
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["role"], "system");
}

#[test]
fn chat_message_with_refusal_serialises() {
    let msg = ChatMessage {
        role: "assistant".into(),
        content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
        refusal: Some("I cannot do that.".into()),
    };
    let v = serde_json::to_value(&msg).unwrap();
    assert_eq!(v["refusal"], "I cannot do that.");
    assert!(v.get("content").is_none());
}

// ---------------------------------------------------------------------------
// MessageContent — text vs parts
// ---------------------------------------------------------------------------

#[test]
fn message_content_text_serialises_as_string() {
    let content = MessageContent::Text("hello world".into());
    let v = serde_json::to_value(&content).unwrap();
    assert_eq!(v, json!("hello world"));
}

#[test]
fn message_content_parts_text_only() {
    let content = MessageContent::Parts(vec![ContentPart::Text {
        text: "What is this?".into(),
    }]);
    let v = serde_json::to_value(&content).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["type"], "text");
    assert_eq!(arr[0]["text"], "What is this?");
}

#[test]
fn message_content_image_url_part() {
    let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://example.com/img.png".into(),
            detail: Some(ImageDetail::High),
        },
    }]);
    let v = serde_json::to_value(&content).unwrap();
    let part = &v.as_array().unwrap()[0];
    assert_eq!(part["type"], "image_url");
    assert_eq!(part["image_url"]["url"], "https://example.com/img.png");
    assert_eq!(part["image_url"]["detail"], "high");
}

#[test]
fn message_content_image_url_no_detail_omitted() {
    let content = MessageContent::Parts(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://example.com/img.png".into(),
            detail: None,
        },
    }]);
    let v = serde_json::to_value(&content).unwrap();
    let img = &v.as_array().unwrap()[0]["image_url"];
    assert!(img.get("detail").is_none());
}

#[test]
fn message_content_audio_part() {
    let content = MessageContent::Parts(vec![ContentPart::InputAudio {
        input_audio: InputAudio {
            data: "base64data".into(),
            format: AudioInputFormat::Wav,
        },
    }]);
    let v = serde_json::to_value(&content).unwrap();
    let part = &v.as_array().unwrap()[0];
    assert_eq!(part["type"], "input_audio");
    assert_eq!(part["input_audio"]["format"], "wav");
}

#[test]
fn message_content_mixed_parts() {
    let content = MessageContent::Parts(vec![
        ContentPart::Text {
            text: "Describe:".into(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/x.jpg".into(),
                detail: Some(ImageDetail::Auto),
            },
        },
    ]);
    let v = serde_json::to_value(&content).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["type"], "text");
    assert_eq!(arr[1]["type"], "image_url");
}

// ---------------------------------------------------------------------------
// ToolChoice
// ---------------------------------------------------------------------------

#[test]
fn tool_choice_mode_none() {
    let tc = ToolChoice::Mode("none".into());
    let v = serde_json::to_value(&tc).unwrap();
    assert_eq!(v, json!("none"));
}

#[test]
fn tool_choice_mode_auto() {
    let v = serde_json::to_value(ToolChoice::Mode("auto".into())).unwrap();
    assert_eq!(v, json!("auto"));
}

#[test]
fn tool_choice_mode_required() {
    let v = serde_json::to_value(ToolChoice::Mode("required".into())).unwrap();
    assert_eq!(v, json!("required"));
}

#[test]
fn tool_choice_named_serialises() {
    let tc = ToolChoice::Named(NamedToolChoice {
        kind: "function".into(),
        function: NamedToolChoiceFunction {
            name: "get_weather".into(),
        },
    });
    let v = serde_json::to_value(&tc).unwrap();
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "get_weather");
}

// ---------------------------------------------------------------------------
// StopSequence
// ---------------------------------------------------------------------------

#[test]
fn stop_single_string() {
    let s = StopSequence::One("STOP".into());
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v, json!("STOP"));
}

#[test]
fn stop_array() {
    let s = StopSequence::Many(vec!["END".into(), "DONE".into()]);
    let v = serde_json::to_value(&s).unwrap();
    assert_eq!(v, json!(["END", "DONE"]));
}

// ---------------------------------------------------------------------------
// ResponseFormat
// ---------------------------------------------------------------------------

#[test]
fn response_format_text() {
    let v = serde_json::to_value(ResponseFormat::Text).unwrap();
    assert_eq!(v["type"], "text");
}

#[test]
fn response_format_json_object() {
    let v = serde_json::to_value(ResponseFormat::JsonObject).unwrap();
    assert_eq!(v["type"], "json_object");
}

#[test]
fn response_format_json_schema() {
    let rf = ResponseFormat::JsonSchema {
        json_schema: JsonSchemaFormat {
            name: "my_schema".into(),
            description: Some("A schema".into()),
            schema: json!({"type": "object"}),
            strict: Some(true),
        },
    };
    let v = serde_json::to_value(&rf).unwrap();
    assert_eq!(v["type"], "json_schema");
    assert_eq!(v["json_schema"]["name"], "my_schema");
    assert_eq!(v["json_schema"]["strict"], true);
}

// ---------------------------------------------------------------------------
// ChatCompletionRequest — optional fields omitted
// ---------------------------------------------------------------------------

#[test]
fn request_minimal_omits_all_optional() {
    let req =
        ChatCompletionRequest::new("gpt-5.4-nano-2026-03-17".into(), vec![ChatMessage::text("user", "hi")], None);
    let v = serde_json::to_value(&req).unwrap();
    // Required
    assert_eq!(v["model"], "gpt-5.4-nano-2026-03-17");
    assert!(v["messages"].is_array());
    // All optional fields must be absent
    for field in &[
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "n",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "logprobs",
        "top_logprobs",
        "seed",
        "store",
        "service_tier",
        "stream",
        "stream_options",
        "reasoning_effort",
    ] {
        assert!(
            v.get(field).is_none(),
            "field '{field}' should be absent when None"
        );
    }
}

#[test]
fn request_with_sampling_params_present() {
    let mut req =
        ChatCompletionRequest::new("gpt-5.4-nano-2026-03-17".into(), vec![ChatMessage::text("user", "hi")], None);
    req.temperature = Some(0.7);
    req.top_p = Some(0.9);
    req.max_completion_tokens = Some(512);
    req.presence_penalty = Some(0.1);
    req.frequency_penalty = Some(0.2);
    req.seed = Some(42);

    let v = serde_json::to_value(&req).unwrap();
    assert!((v["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-5);
    assert!((v["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-5);
    assert_eq!(v["max_completion_tokens"], 512);
    assert_eq!(v["seed"], 42);
}

#[test]
fn request_with_tools_present() {
    use superglue::tools::ToolSpec;
    let tool = ChatTool::from(ToolSpec {
        name: "search".into(),
        description: Some("Search the web".into()),
        parameters_schema: json!({"type": "object"}),
                static_tool: false,
        });
    let req = ChatCompletionRequest::new(
        "gpt-5.4-nano-2026-03-17".into(),
        vec![ChatMessage::text("user", "search for Rust")],
        Some(vec![tool]),
    );
    let v = serde_json::to_value(&req).unwrap();
    let tools = v["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "search");
}

#[test]
fn request_stream_true_serialises() {
    let mut req = ChatCompletionRequest::new(
        "gpt-5.4-nano-2026-03-17-mini".into(),
        vec![ChatMessage::text("user", "hi")],
        None,
    );
    req.stream = Some(true);
    req.stream_options = Some(StreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    });
    let v = serde_json::to_value(&req).unwrap();
    assert_eq!(v["stream"], true);
    assert_eq!(v["stream_options"]["include_usage"], true);
    assert!(v["stream_options"].get("include_obfuscation").is_none());
}

// ---------------------------------------------------------------------------
// ChatCompletionResponse — deserialization
// ---------------------------------------------------------------------------

#[test]
fn response_deserializes_text_completion() {
    let raw = json!({
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1700000000u64,
        "model": "gpt-5.4-nano-2026-03-17-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    });
    let resp: superglue::openai::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(resp.id, "chatcmpl-abc");
    assert_eq!(resp.model, "gpt-5.4-nano-2026-03-17-mini");
    assert_eq!(resp.choices.len(), 1);
    let choice = &resp.choices[0];
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    let content = choice.message.content.as_ref().unwrap().as_text().unwrap();
    assert_eq!(content, "Hello!");
    let usage = resp.usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn response_deserializes_tool_call() {
    let raw = json!({
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "model": "gpt-5.4-nano-2026-03-17",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let resp: superglue::openai::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
    let msg = &resp.choices[0].message;
    assert!(msg.content.is_none());
    let tcs = msg.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].id, "call_xyz");
    assert_eq!(tcs[0].function.name, "get_weather");
    assert!(tcs[0].function.arguments.contains("Paris"));
}

#[test]
fn response_missing_usage_is_none() {
    let raw = json!({
        "id": "chatcmpl-nousage",
        "object": "chat.completion",
        "model": "gpt-5.4-nano-2026-03-17",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }]
    });
    let resp: superglue::openai::ChatCompletionResponse = serde_json::from_value(raw).unwrap();
    assert!(resp.usage.is_none());
}

// ---------------------------------------------------------------------------
// ChatCompletionChunk — streaming deserialization
// ---------------------------------------------------------------------------

#[test]
fn chunk_content_delta_deserializes() {
    let raw = json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000u64,
        "model": "gpt-5.4-nano-2026-03-17-mini",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": "Hello"},
            "finish_reason": null
        }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    assert_eq!(chunk.choices.len(), 1);
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
    assert!(chunk.choices[0].finish_reason.is_none());
}

#[test]
fn chunk_final_with_finish_reason_and_usage() {
    let raw = json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000u64,
        "model": "gpt-5.4-nano-2026-03-17-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    let u = chunk.usage.as_ref().unwrap();
    assert_eq!(u.total_tokens, 20);
}

#[test]
fn chunk_tool_call_delta_deserializes() {
    let raw = json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000u64,
        "model": "gpt-5.4-nano-2026-03-17-mini",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{\"x\":"}
                }]
            },
            "finish_reason": null
        }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    let tcs = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].id.as_deref(), Some("call_abc"));
    assert_eq!(
        tcs[0].function.as_ref().unwrap().name.as_deref(),
        Some("echo")
    );
}

#[test]
fn chunk_empty_delta_deserializes() {
    let raw = json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000u64,
        "model": "gpt-5.4-nano-2026-03-17-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    assert!(chunk.choices[0].delta.content.is_none());
    assert!(chunk.choices[0].delta.role.is_none());
    assert!(chunk.choices[0].delta.tool_calls.is_none());
}

// ---------------------------------------------------------------------------
// ChatTool (from ToolSpec)
// ---------------------------------------------------------------------------

#[test]
fn chat_tool_from_spec_with_description() {
    use superglue::tools::ToolSpec;
    let spec = ToolSpec {
        name: "get_time".into(),
        description: Some("Returns the current UTC time.".into()),
        parameters_schema: json!({"type": "object", "properties": {}}),
        static_tool: false,
    };
    let tool = ChatTool::from(spec);
    let v = serde_json::to_value(&tool).unwrap();
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "get_time");
    assert_eq!(
        v["function"]["description"],
        "Returns the current UTC time."
    );
    assert!(v["function"]["strict"].is_null() || v["function"].get("strict").is_none());
}

#[test]
fn chat_tool_from_spec_no_description_omitted() {
    use superglue::tools::ToolSpec;
    let spec = ToolSpec {
        name: "noop".into(),
        description: None,
        parameters_schema: json!({}),
                static_tool: false,
        };
    let tool = ChatTool::from(spec);
    let v = serde_json::to_value(&tool).unwrap();
    assert!(v["function"].get("description").is_none());
}

// ---------------------------------------------------------------------------
// ToolCall (in response)
// ---------------------------------------------------------------------------

#[test]
fn tool_call_round_trips() {
    let tc = ToolCall {
        id: "call_1".into(),
        kind: "function".into(),
        function: FunctionCall {
            name: "add".into(),
            arguments: "{\"a\":1,\"b\":2}".into(),
        },
    };
    let v = serde_json::to_value(&tc).unwrap();
    assert_eq!(v["id"], "call_1");
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "add");

    let back: ToolCall = serde_json::from_value(v).unwrap();
    assert_eq!(back.id, "call_1");
    assert_eq!(back.function.arguments, "{\"a\":1,\"b\":2}");
}
