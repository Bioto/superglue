//! Dynamic tool routing — GlueLLM-compatible `request_tools` meta-tool.

use serde_json::json;

use crate::openai::{ChatTool, ToolCall};

use super::types::ToolSpec;

/// Router meta-tool name (GlueLLM parity).
pub const ROUTER_TOOL_NAME: &str = "request_tools";

/// Programmatic tool-calling meta-tool. The model writes JavaScript that
/// invokes routed tools via `tools.<name>(args)`.
pub const CODE_TOOL_NAME: &str = "code";

/// How tools are exposed to the model each completion round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    /// Every registered tool schema is sent on every LLM call.
    #[default]
    Standard,
    /// Only a router tool (+ static tools) until routing selects a subset.
    Dynamic,
    /// Router returns matched tool schemas as data; the model then calls `code`.
    Code,
}

impl ToolMode {
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "dynamic" => Self::Dynamic,
            "code" | "programmatic" => Self::Code,
            _ => Self::Standard,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Dynamic => "dynamic",
            Self::Code => "code",
        }
    }

    /// Whether the first LLM round should see `request_tools` instead of the full catalog.
    #[must_use]
    pub const fn uses_router(self) -> bool {
        matches!(self, Self::Dynamic | Self::Code)
    }
}

/// Per-request active tool set for dynamic routing.
#[derive(Debug, Clone)]
pub struct ActiveToolSet {
    mode: ToolMode,
    static_specs: Vec<ToolSpec>,
    dynamic_specs: Vec<ToolSpec>,
    active_specs: Vec<ToolSpec>,
    /// Tools the `code` runtime may invoke (matched dynamic + static).
    callable_specs: Vec<ToolSpec>,
    llm_chat_tools: Vec<ChatTool>,
    router_spec: Option<ToolSpec>,
    pub routed: bool,
}

fn build_llm_chat_tools(specs: &[ToolSpec]) -> Vec<ChatTool> {
    specs.iter().map(|s| ChatTool::from(s.clone())).collect()
}

impl ActiveToolSet {
    /// Build from all registered specs and the configured tool mode.
    #[must_use]
    pub fn new(all_specs: Vec<ToolSpec>, mode: ToolMode) -> Self {
        if !mode.uses_router() || all_specs.is_empty() {
            return Self {
                mode,
                static_specs: Vec::new(),
                dynamic_specs: Vec::new(),
                active_specs: all_specs.clone(),
                callable_specs: Vec::new(),
                llm_chat_tools: build_llm_chat_tools(&all_specs),
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

        let router_spec = Some(match mode {
            ToolMode::Code => build_code_router_tool_spec(&dynamic_specs),
            _ => build_router_tool_spec(&dynamic_specs),
        });
        let mut active_specs = Vec::with_capacity(1 + static_specs.len());
        if let Some(ref router) = router_spec {
            active_specs.push(router.clone());
        }
        active_specs.extend(static_specs.clone());

        Self {
            mode,
            static_specs,
            dynamic_specs,
            active_specs: active_specs.clone(),
            callable_specs: Vec::new(),
            llm_chat_tools: build_llm_chat_tools(&active_specs),
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

    /// Pre-built OpenAI `tools[]` entries for the active set (avoids per-request cloning).
    #[must_use]
    pub fn chat_tools_for_llm(&self) -> Option<&[ChatTool]> {
        if self.llm_chat_tools.is_empty() {
            None
        } else {
            Some(&self.llm_chat_tools)
        }
    }

    pub fn apply_route(&mut self, matched: Vec<ToolSpec>) {
        if self.mode == ToolMode::Code {
            self.apply_code_route(matched);
            return;
        }
        self.active_specs = matched;
        self.active_specs.extend(self.static_specs.clone());
        self.llm_chat_tools = build_llm_chat_tools(&self.active_specs);
        self.router_spec = None;
        self.routed = true;
    }

    /// After routing in [`ToolMode::Code`], expose only `code` (+ static tools).
    pub fn apply_code_route(&mut self, matched: Vec<ToolSpec>) {
        self.callable_specs = matched;
        self.callable_specs.extend(self.static_specs.clone());
        let mut active_specs = vec![build_code_tool_spec()];
        active_specs.extend(self.static_specs.clone());
        self.active_specs = active_specs;
        self.llm_chat_tools = build_llm_chat_tools(&self.active_specs);
        self.router_spec = None;
        self.routed = true;
    }

    /// JSON catalog returned as the `request_tools` result in code mode.
    #[must_use]
    pub fn route_catalog(&self, matched: &[ToolSpec]) -> serde_json::Value {
        build_route_catalog(matched, &self.static_specs)
    }

    /// Names the `code` runtime may invoke after a successful code-mode route.
    #[must_use]
    pub fn code_allowlist(&self) -> std::collections::HashSet<String> {
        self.callable_specs.iter().map(|s| s.name.clone()).collect()
    }

    #[must_use]
    pub fn mode(&self) -> ToolMode {
        self.mode
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

/// Build the `code` tool schema shown after routing in [`ToolMode::Code`].
#[must_use]
pub fn build_code_tool_spec() -> ToolSpec {
    ToolSpec {
        name: CODE_TOOL_NAME.to_string(),
        description: Some(
            "Run JavaScript that calls the routed tools. Use `tools.<name>(args)` \
             synchronously (or `tools.call(name, args)`). Return the reduced JSON \
             the user needs — do not dump raw tool payloads unless required. \
             `code` and `request_tools` cannot be called from the script."
                .into(),
        ),
        parameters_schema: json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "JavaScript source. Call tools.<name>({...}) and `return` a JSON value."
                }
            },
            "required": ["source"]
        }),
        static_tool: true,
    }
}

/// Catalog of matched tools + argument schemas returned by `request_tools` in code mode.
#[must_use]
pub fn build_route_catalog(matched: &[ToolSpec], static_tools: &[ToolSpec]) -> serde_json::Value {
    fn spec_entry(spec: &ToolSpec) -> serde_json::Value {
        json!({
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters_schema,
        })
    }
    json!({
        "tools": matched.iter().map(spec_entry).collect::<Vec<_>>(),
        "static_tools": static_tools.iter().map(spec_entry).collect::<Vec<_>>(),
        "invoke": "Call the `code` tool with JavaScript. Use tools.<name>(args) synchronously and return a JSON value. Do not call request_tools again."
    })
}

/// Router schema for [`ToolMode::Code`]: discover tools, then write `code`.
#[must_use]
pub fn build_code_router_tool_spec(dynamic_tools: &[ToolSpec]) -> ToolSpec {
    let names: Vec<&str> = dynamic_tools.iter().map(|t| t.name.as_str()).collect();
    let names_str = names.join(", ");
    ToolSpec {
        name: ROUTER_TOOL_NAME.to_string(),
        description: Some(format!(
            "Call this first to discover which tools are needed for the user's request. \
             Available tools: {names_str}. Pass the user's query (or a short summary) as the 'query' argument. \
             You will receive each matched tool's name, description, and argument schema, \
             plus a single `code` tool. Write JavaScript that calls tools.<name>(args) \
             instead of invoking those tools directly."
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

/// Id of the first `request_tools` call (for the code-mode catalog tool result).
#[must_use]
pub fn router_call_id_from_calls(tool_calls: &[ToolCall]) -> Option<String> {
    tool_calls
        .iter()
        .find(|tc| tc.kind == "function" && tc.function.name == ROUTER_TOOL_NAME)
        .map(|tc| tc.id.clone())
}

/// Return true when any tool call targets the `code` meta-tool.
#[must_use]
pub fn is_code_call(tool_calls: &[ToolCall]) -> bool {
    tool_calls
        .iter()
        .filter(|tc| tc.kind == "function")
        .any(|tc| tc.function.name == CODE_TOOL_NAME)
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

    #[test]
    fn tool_mode_parse_accepts_code_aliases() {
        assert_eq!(ToolMode::parse("code"), ToolMode::Code);
        assert_eq!(ToolMode::parse("programmatic"), ToolMode::Code);
        assert_eq!(ToolMode::parse("dynamic"), ToolMode::Dynamic);
        assert_eq!(ToolMode::parse("standard"), ToolMode::Standard);
    }

    #[test]
    fn code_mode_starts_with_router_and_static() {
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
        let set = ActiveToolSet::new(specs, ToolMode::Code);
        assert!(set.has_router());
        let names: Vec<_> = set
            .specs_for_llm()
            .unwrap()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec![ROUTER_TOOL_NAME, "pinned"]);
        assert!(
            set.specs_for_llm()
                .unwrap()
                .iter()
                .any(|s| s.description.as_deref().is_some_and(|d| d.contains("code")))
        );
    }

    #[test]
    fn apply_code_route_exposes_code_and_static_only() {
        let specs = vec![
            ToolSpec {
                name: "pinned".into(),
                description: None,
                parameters_schema: json!({}),
                static_tool: true,
            },
            ToolSpec {
                name: "get_weather".into(),
                description: Some("Weather".into()),
                parameters_schema: json!({"type":"object"}),
                static_tool: false,
            },
        ];
        let mut set = ActiveToolSet::new(specs.clone(), ToolMode::Code);
        set.apply_route(vec![specs[1].clone()]);
        let names: Vec<_> = set
            .specs_for_llm()
            .unwrap()
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names, vec![CODE_TOOL_NAME, "pinned"]);
        assert!(set.code_allowlist().contains("get_weather"));
        assert!(set.code_allowlist().contains("pinned"));
        let catalog = set.route_catalog(&[specs[1].clone()]);
        assert_eq!(catalog["tools"][0]["name"], "get_weather");
        assert_eq!(catalog["static_tools"][0]["name"], "pinned");
    }
}
