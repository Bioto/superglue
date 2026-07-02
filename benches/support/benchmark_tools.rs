//! Nine-tool fixture matching GlueLLM's dynamic routing benchmark.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use superglue::tools::{Tool, ToolInvokeError, ToolRegistry, ToolSpec};

struct JsonTool {
    spec: ToolSpec,
    result: Value,
}

#[async_trait]
impl Tool for JsonTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, _args: Value) -> Result<Value, ToolInvokeError> {
        Ok(self.result.clone())
    }
}

fn dynamic_tool(name: &str, description: &str, result: Value) -> Arc<dyn Tool> {
    Arc::new(JsonTool {
        spec: ToolSpec {
            name: name.into(),
            description: Some(description.into()),
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "city": { "type": "string" },
                    "expression": { "type": "string" },
                    "from": { "type": "string" },
                    "to": { "type": "string" },
                    "text": { "type": "string" },
                    "target_lang": { "type": "string" }
                }
            }),
            static_tool: false,
        },
        result,
    })
}

/// All dynamic tool names (excluding pinned static); used by wiremock fixtures.
#[allow(dead_code)]
pub const DYNAMIC_TOOL_NAMES: &[&str] = &[
    "get_weather",
    "get_forecast",
    "search_flights",
    "book_hotel",
    "calculate",
    "get_exchange_rate",
    "translate_text",
    "get_country_info",
];

/// Register the full 9-tool GlueLLM-style set on a registry (wiremock / tests).
#[allow(dead_code)]
pub async fn register_benchmark_tools(registry: &ToolRegistry) -> Result<(), ToolInvokeError> {
    for tool in all_benchmark_tools() {
        registry.register(tool).await?;
    }
    Ok(())
}

/// Tool instances for live client registration.
#[must_use]
pub fn all_benchmark_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        dynamic_tool(
            "get_weather",
            "Get current weather for a city",
            json!({"city": "Paris", "temp_f": 72, "condition": "sunny"}),
        ),
        dynamic_tool(
            "get_forecast",
            "Get multi-day weather forecast for a city",
            json!({"city": "Paris", "days": [{"high_f": 75}, {"high_f": 73}, {"high_f": 71}]}),
        ),
        dynamic_tool(
            "search_flights",
            "Search flights between two cities",
            json!({"from": "NYC", "to": "Paris", "price_usd": 650}),
        ),
        dynamic_tool(
            "book_hotel",
            "Book a hotel in a city",
            json!({"city": "Paris", "nights": 3, "confirmed": true}),
        ),
        dynamic_tool(
            "calculate",
            "Evaluate a math expression",
            json!({"expression": "avg", "result": 73.0}),
        ),
        dynamic_tool(
            "get_exchange_rate",
            "Get currency exchange rate",
            json!({"from": "USD", "to": "EUR", "rate": 0.92}),
        ),
        dynamic_tool(
            "translate_text",
            "Translate text to a target language",
            json!({"text": "hello", "target_lang": "fr", "translation": "bonjour"}),
        ),
        dynamic_tool(
            "get_country_info",
            "Get country facts",
            json!({"country": "France", "capital": "Paris", "currency": "EUR"}),
        ),
        Arc::new(JsonTool {
            spec: ToolSpec {
                name: "get_time".into(),
                description: Some("Return current UTC time (always available)".into()),
                parameters_schema: json!({"type": "object", "properties": {}}),
                static_tool: true,
            },
            result: json!({"time": "12:00Z"}),
        }),
    ]
}
