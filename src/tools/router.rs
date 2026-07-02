//! Dynamic tool routing — GlueLLM-compatible `request_tools` meta-tool.

use serde_json::json;

use crate::openai::ToolCall;

use super::types::ToolSpec;

/// Router meta-tool name (GlueLLM parity).
pub const ROUTER_TOOL_NAME: &str = "request_tools";

/// How tools are exposed to the model each completion round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    /// Every registered tool schema is sent on every LLM call.
    #[default]
    Standard,
    /// Only a router tool (+ static tools) until routing selects a subset.
    Dynamic,
}

/// Per-request active tool set for dynamic routing.
#[derive(Debug, Clone)]
pub struct ActiveToolSet {
    static_specs: Vec<ToolSpec>,
    dynamic_specs: Vec<ToolSpec>,
    active_specs: Vec<ToolSpec>,
    router_spec: Option<ToolSpec>,
    pub routed: bool,
}

impl ActiveToolSet {
    /// Build from all registered specs and the configured tool mode.
    #[must_use]
    pub fn new(all_specs: Vec<ToolSpec>, mode: ToolMode) -> Self {
        if mode != ToolMode::Dynamic || all_specs.is_empty() {
            return Self {
                static_specs: Vec::new(),
                dynamic_specs: Vec::new(),
                active_specs: all_specs,
                router_spec: None,
                routed: false,
            };
        }

        let static_specs: Vec<_> = all_specs
            .iter()
            .filter(|s| s.static_tool)
            .cloned()
            .collect();
        let dynamic_specs: Vec<_> = all_specs
            .iter()
            .filter(|s| !s.static_tool)
            .cloned()
            .collect();

        let router_spec = Some(build_router_tool_spec(&dynamic_specs));
        let mut active_specs = Vec::with_capacity(1 + static_specs.len());
        if let Some(ref router) = router_spec {
            active_specs.push(router.clone());
        }
        active_specs.extend(static_specs.clone());

        Self {
            static_specs,
            dynamic_specs,
            active_specs,
            router_spec,
            routed: false,
        }
    }

    #[must_use]
    pub fn specs_for_llm(&self) -> Option<&[ToolSpec]> {
        if self.active_specs.is_empty() {
            None
        } else {
            Some(&self.active_specs)
        }
    }

    pub fn apply_route(&mut self, matched: Vec<ToolSpec>) {
        self.active_specs = matched;
        self.active_specs.extend(self.static_specs.clone());
        self.router_spec = None;
        self.routed = true;
    }

    #[must_use]
    pub fn dynamic_specs(&self) -> &[ToolSpec] {
        &self.dynamic_specs
    }

    #[must_use]
    pub fn static_specs(&self) -> &[ToolSpec] {
        &self.static_specs
    }

    #[must_use]
    pub fn has_router(&self) -> bool {
        self.router_spec.is_some()
    }
}

/// Build the synthetic router tool schema shown to the model in dynamic mode.
#[must_use]
pub fn build_router_tool_spec(dynamic_tools: &[ToolSpec]) -> ToolSpec {
    let names: Vec<&str> = dynamic_tools.iter().map(|t| t.name.as_str()).collect();
    let names_str = names.join(", ");
    ToolSpec {
        name: ROUTER_TOOL_NAME.to_string(),
        description: Some(format!(
            "Call this first to discover which tools are needed for the user's request. \
             Available tools: {names_str}. Pass the user's query (or a short summary) as the 'query' argument. \
             After calling, you will receive the relevant tool schemas."
        )),
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's request or a short summary of what tools are needed."
                }
            },
            "required": ["query"]
        }),
        static_tool: false,
    }
}

/// Return true when any tool call targets the router meta-tool.
#[must_use]
pub fn is_router_call(tool_calls: &[ToolCall]) -> bool {
    tool_calls
        .iter()
        .filter(|tc| tc.kind == "function")
        .any(|tc| tc.function.name == ROUTER_TOOL_NAME)
}

/// Extract the routing query from the first router tool call.
#[must_use]
pub fn router_query_from_calls(tool_calls: &[ToolCall]) -> Option<String> {
    tool_calls
        .iter()
        .filter(|tc| tc.kind == "function" && tc.function.name == ROUTER_TOOL_NAME)
        .find_map(|tc| {
            let raw = tc.function.arguments.trim();
            if raw.is_empty() {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(raw)
                .ok()
                .and_then(|v| v.get("query").and_then(|q| q.as_str()).map(str::to_string))
        })
}

/// Default fast model for routing when `tool_route_model` is unset.
pub const DEFAULT_TOOL_ROUTE_MODEL: &str = "gpt-5.4-nano-2026-03-17-mini";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_router_spec_lists_dynamic_tools() {
        let specs = vec![
            ToolSpec {
                name: "get_weather".into(),
                description: Some("Weather".into()),
                parameters_schema: json!({}),
                static_tool: false,
            },
            ToolSpec {
                name: "always_on".into(),
                description: None,
                parameters_schema: json!({}),
                static_tool: true,
            },
        ];
        let router = build_router_tool_spec(&[specs[0].clone()]);
        assert_eq!(router.name, ROUTER_TOOL_NAME);
        assert!(router.description.unwrap().contains("get_weather"));
    }

    #[test]
    fn active_set_dynamic_starts_with_router_and_static() {
        let specs = vec![
            ToolSpec {
                name: "pinned".into(),
                description: None,
                parameters_schema: json!({}),
                static_tool: true,
            },
            ToolSpec {
                name: "dynamic_a".into(),
                description: None,
                parameters_schema: json!({}),
                static_tool: false,
            },
        ];
        let set = ActiveToolSet::new(specs, ToolMode::Dynamic);
        assert!(set.has_router());
        assert_eq!(set.specs_for_llm().unwrap().len(), 2);
        assert_eq!(set.specs_for_llm().unwrap()[0].name, ROUTER_TOOL_NAME);
        assert_eq!(set.specs_for_llm().unwrap()[1].name, "pinned");
    }

    #[test]
    fn is_router_call_detects_request_tools() {
        let tc = ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: crate::openai::FunctionCall {
                name: ROUTER_TOOL_NAME.into(),
                arguments: r#"{"query":"weather"}"#.into(),
            },
        };
        assert!(is_router_call(&[tc]));
    }
}
