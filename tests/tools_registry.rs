//! Integration tests for [`superglue::tools`].

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use superglue::tools::{
    PlanStep, RunEvent, Tool, ToolInvocation, ToolInvokeError, ToolRegistry, ToolSpec, run_plan,
};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: None,
            parameters_schema: json!({"type": "object"}),
        }
    }

    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

struct FailTool;

#[async_trait]
impl Tool for FailTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fail".to_string(),
            description: None,
            parameters_schema: json!({}),
        }
    }

    async fn call(&self, _arguments: Value) -> Result<Value, ToolInvokeError> {
        Err(ToolInvokeError::handler(
            "intentional failure",
            Some("demo".to_string()),
        ))
    }
}

#[tokio::test]
async fn invoke_unknown_tool() {
    let reg = ToolRegistry::new();
    let err = reg.invoke("missing", json!({})).await.unwrap_err();
    assert!(matches!(err, ToolInvokeError::UnknownTool { .. }));
}

#[tokio::test]
async fn register_duplicate_rejected() {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let err = reg.register(Arc::new(EchoTool)).await.unwrap_err();
    assert!(matches!(
        err,
        ToolInvokeError::DuplicateRegistration { name } if name == "echo"
    ));
}

#[tokio::test]
async fn invoke_success() {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let out = reg.invoke("echo", json!({"a": 1})).await.unwrap();
    assert_eq!(out, json!({"echo": {"a": 1}}));
}

#[tokio::test]
async fn handler_error_surfaces() {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(FailTool)).await.unwrap();
    let err = reg.invoke("fail", json!({})).await.unwrap_err();
    assert!(matches!(err, ToolInvokeError::HandlerFailed { .. }));
    let j = err.to_json();
    assert_eq!(j["error"], "intentional failure");
    assert_eq!(j["code"], "demo");
}

#[tokio::test]
async fn harness_plan_round_trip() {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let steps = vec![
        PlanStep::ToolCall(ToolInvocation {
            tool_name: "echo".into(),
            arguments: json!({"city": "Paris"}),
        }),
        PlanStep::Done {
            message: "done".into(),
        },
    ];
    let events = run_plan(&reg, &steps).await.unwrap();
    assert_eq!(
        events,
        vec![
            RunEvent::ToolResult {
                tool_name: "echo".into(),
                result: json!({"echo": {"city": "Paris"}}),
            },
            RunEvent::Done {
                message: "done".into(),
            },
        ]
    );
}

#[tokio::test]
async fn harness_aborts_on_tool_error() {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(FailTool)).await.unwrap();
    let steps = vec![PlanStep::ToolCall(ToolInvocation {
        tool_name: "fail".into(),
        arguments: json!({}),
    })];
    let err = run_plan(&reg, &steps).await.unwrap_err();
    assert!(matches!(err, ToolInvokeError::HandlerFailed { .. }));
}
