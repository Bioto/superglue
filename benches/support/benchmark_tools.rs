//! Nine-tool fixture matching GlueLLM's dynamic routing benchmark.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
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

/// Size of the synthetic `raw` dump attached to each fat-payload tool result.
///
/// ~40k chars per tool is large enough that a 6-tool chain exceeds ~50k tokens
/// in the next LLM round if dumps are stuffed into chat (and exceeds the
/// default 32k-char tool-result cap, so benches raise `tool_result_max_chars`).
#[allow(dead_code)]
pub const FAT_RAW_CHARS: usize = 40_000;

/// Same tools with a bulky `raw` field so programmatic calling can drop intermediates.
#[allow(dead_code)]
pub async fn fat_benchmark_tools(raw_chars: usize) -> Vec<Arc<dyn Tool>> {
    let blob = "x".repeat(raw_chars);
    let mut tools = Vec::new();
    for tool in all_benchmark_tools() {
        let spec = tool.spec();
        let mut result = tool
            .call(json!({}))
            .await
            .expect("benchmark fixture tools do not fail");
        if !spec.static_tool
            && let Some(obj) = result.as_object_mut()
        {
            obj.insert("raw".into(), json!(blob));
        }
        tools.push(Arc::new(JsonTool { spec, result }) as Arc<dyn Tool>);
    }
    tools
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fat_payloads_skip_static_get_time() {
        let tools = fat_benchmark_tools(128).await;
        let time = tools
            .iter()
            .find(|t| t.spec().name == "get_time")
            .expect("get_time");
        let value = time.call(json!({})).await.expect("call");
        assert_eq!(value["time"], "12:00Z");
        assert!(value.get("raw").is_none());

        let weather = tools
            .iter()
            .find(|t| t.spec().name == "get_weather")
            .expect("get_weather");
        let value = weather.call(json!({})).await.expect("call");
        assert_eq!(value["raw"].as_str().unwrap().len(), 128);
    }
}

/// Register [`fat_benchmark_tools`] on a registry (wiremock / tests).
#[allow(dead_code)]
pub async fn register_fat_benchmark_tools(
    registry: &ToolRegistry,
    raw_chars: usize,
) -> Result<(), ToolInvokeError> {
    for tool in fat_benchmark_tools(raw_chars).await {
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
            "Get current weather for a city. Returns city, temp_f, condition.",
            json!({"city": "Paris", "temp_f": 72, "condition": "sunny"}),
        ),
        dynamic_tool(
            "get_forecast",
            "Get multi-day weather forecast for a city. Returns city and days[].high_f.",
            json!({"city": "Paris", "days": [{"high_f": 75}, {"high_f": 73}, {"high_f": 71}]}),
        ),
        dynamic_tool(
            "search_flights",
            "Search flights between two cities. Returns from, to, price_usd.",
            json!({"from": "NYC", "to": "Paris", "price_usd": 650}),
        ),
        dynamic_tool(
            "book_hotel",
            "Book a hotel in a city. Returns city, nights, confirmed.",
            json!({"city": "Paris", "nights": 3, "confirmed": true}),
        ),
        dynamic_tool(
            "calculate",
            "Evaluate a math expression. Returns expression, result.",
            json!({"expression": "avg", "result": 73.0}),
        ),
        dynamic_tool(
            "get_exchange_rate",
            "Get currency exchange rate. Returns from, to, rate.",
            json!({"from": "USD", "to": "EUR", "rate": 0.92}),
        ),
        dynamic_tool(
            "translate_text",
            "Translate text to a target language. Returns text, target_lang, translation.",
            json!({"text": "hello", "target_lang": "fr", "translation": "bonjour"}),
        ),
        dynamic_tool(
            "get_country_info",
            "Get country facts. Returns country, capital, currency.",
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
