//! Dynamic tool routing and context condensing (GlueLLM parity).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use async_trait::async_trait;
use serde_json::json;
use superglue::chat::{ChatOptions, complete_with_tools};
use superglue::context::SummarizeContextConfig;
use superglue::events::{ProcessEventKind, StatusEmitter, StatusSubscriber};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{CODE_TOOL_NAME, ROUTER_TOOL_NAME, Tool, ToolMode, ToolRegistry, ToolSpec};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_weather".into(),
            description: Some("Get weather for a city".into()),
            parameters_schema: json!({"type":"object","properties":{"city":{"type":"string"}}}),
            static_tool: false,
        }
    }

    async fn call(
        &self,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({"city": args["city"], "temp": 72}))
    }
}

struct CalcTool;

#[async_trait]
impl Tool for CalcTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "calculate".into(),
            description: Some("Evaluate math".into()),
            parameters_schema: json!({"type":"object"}),
            static_tool: false,
        }
    }

    async fn call(
        &self,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({"result": 4}))
    }
}

struct PinnedTool;

#[async_trait]
impl Tool for PinnedTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "get_time".into(),
            description: Some("Current UTC time".into()),
            parameters_schema: json!({"type":"object"}),
            static_tool: true,
        }
    }

    async fn call(
        &self,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({"time": "12:00Z"}))
    }
}

fn router_call_response() -> serde_json::Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_router",
                    "type": "function",
                    "function": {
                        "name": ROUTER_TOOL_NAME,
                        "arguments": "{\"query\":\"weather in Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn route_resolve_response() -> serde_json::Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "[\"get_weather\"]"
            },
            "finish_reason": "stop"
        }]
    })
}

fn weather_tool_call_response() -> serde_json::Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_w",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn code_tool_call_response() -> serde_json::Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_code",
                    "type": "function",
                    "function": {
                        "name": CODE_TOOL_NAME,
                        "arguments": "{\"source\":\"const w = tools.get_weather({city:\\\"Paris\\\"}); return {temp: w.temp};\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

fn final_text_response() -> serde_json::Value {
    json!({
        "model": "mock",
        "choices": [{
            "message": {"role": "assistant", "content": "It is 72F in Paris."},
            "finish_reason": "stop"
        }]
    })
}

struct EventCollector {
    events: Arc<tokio::sync::Mutex<Vec<superglue::events::ProcessEvent>>>,
}

#[async_trait]
impl StatusSubscriber for EventCollector {
    async fn on_event(&self, event: superglue::events::ProcessEvent) {
        self.events.lock().await.push(event);
    }
}

#[tokio::test]
async fn dynamic_routing_selects_tools_then_executes() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = match i {
                0 => router_call_response(),
                1 => route_resolve_response(),
                2 => weather_tool_call_response(),
                _ => final_text_response(),
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(WeatherTool)).await.unwrap();
    reg.register(Arc::new(CalcTool)).await.unwrap();
    reg.register(Arc::new(PinnedTool)).await.unwrap();

    let events = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let emitter = StatusEmitter::new();
    emitter
        .subscribe(Arc::new(EventCollector {
            events: Arc::clone(&events),
        }))
        .await;

    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        tool_mode: ToolMode::Dynamic,
        tool_route_model: Some("mock".into()),
        status_emitter: Some(Arc::new(emitter)),
        ..Default::default()
    };

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "What's the weather in Paris?")],
        &opts,
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("It is 72F in Paris."));
    let evs = events.lock().await;
    assert!(evs.iter().any(|e| e.kind == ProcessEventKind::ToolRoute));
}

#[tokio::test]
async fn code_mode_returns_catalog_then_runs_javascript() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    let seen_tools = Arc::new(tokio::sync::Mutex::new(Vec::<Vec<String>>::new()));
    let seen_tools2 = Arc::clone(&seen_tools);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&req.body) {
                let names = body
                    .get("tools")
                    .and_then(|t| t.as_array())
                    .map(|tools| {
                        tools
                            .iter()
                            .filter_map(|t| {
                                t.pointer("/function/name")
                                    .or_else(|| t.get("name"))
                                    .and_then(|n| n.as_str())
                                    .map(str::to_string)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if let Ok(mut guard) = seen_tools2.try_lock() {
                    guard.push(names);
                }
            }
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = match i {
                0 => router_call_response(),
                1 => route_resolve_response(),
                2 => code_tool_call_response(),
                _ => final_text_response(),
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(WeatherTool)).await.unwrap();
    reg.register(Arc::new(CalcTool)).await.unwrap();
    reg.register(Arc::new(PinnedTool)).await.unwrap();

    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 8,
        tool_mode: ToolMode::Code,
        tool_route_model: Some("mock".into()),
        ..Default::default()
    };

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "What's the weather in Paris?")],
        &opts,
    )
    .await
    .unwrap();

    assert_eq!(out.content.as_deref(), Some("It is 72F in Paris."));

    let catalog = out
        .messages
        .iter()
        .find(|m| m.role == "tool" && m.name.as_deref() == Some(ROUTER_TOOL_NAME));
    let catalog_text = catalog
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text())
        .expect("request_tools catalog");
    let catalog_json: serde_json::Value = serde_json::from_str(catalog_text).unwrap();
    assert_eq!(catalog_json["tools"][0]["name"], "get_weather");
    assert!(catalog_json["tools"][0]["parameters"].is_object());
    assert_eq!(catalog_json["static_tools"][0]["name"], "get_time");

    let code_result = out
        .messages
        .iter()
        .find(|m| m.role == "tool" && m.name.as_deref() == Some(CODE_TOOL_NAME));
    let code_text = code_result
        .and_then(|m| m.content.as_ref())
        .and_then(|c| c.as_text())
        .expect("code result");
    let code_json: serde_json::Value = serde_json::from_str(code_text).unwrap();
    assert_eq!(code_json["ok"], true);
    assert_eq!(code_json["result"]["temp"], 72);

    let seen = seen_tools.lock().await;
    let after_route = seen
        .iter()
        .find(|names| names.iter().any(|n| n == CODE_TOOL_NAME))
        .expect("LLM request that exposed code");
    assert!(after_route.iter().any(|n| n == CODE_TOOL_NAME));
    assert!(after_route.iter().any(|n| n == "get_time"));
    assert!(!after_route.iter().any(|n| n == "get_weather"));
    assert!(!after_route.iter().any(|n| n == "calculate"));
}

#[tokio::test]
async fn condense_tool_messages_keeps_unprofitable_round() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let body = if i == 0 {
                weather_tool_call_response()
            } else {
                final_text_response()
            };
            ResponseTemplate::new(200).set_body_json(body)
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(WeatherTool)).await.unwrap();

    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "mock".into(),
        max_tool_rounds: 4,
        condense_tool_messages: true,
        ..Default::default()
    };

    let out = complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "Weather in Paris")],
        &opts,
    )
    .await
    .unwrap();

    // A one-call weather round is not cheaper after the [Tool Results] rewrite, so
    // context_ops keeps the original assistant + tool messages.
    assert!(out.messages.iter().any(|m| {
        m.role == "tool"
            && m.content
                .as_ref()
                .and_then(|c| c.as_text())
                .is_some_and(|t| t.contains("72"))
    }));
}

#[test]
fn summarize_context_config_defaults_match_gluellm() {
    let cfg = SummarizeContextConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.threshold, 20);
    assert_eq!(cfg.keep_recent, 6);
}
