//! Deterministic fake assistant plan (no LLM) for testing the tool boundary.

use serde_json::Value;

use super::error::ToolInvokeError;
use super::registry::ToolRegistry;
use super::types::ToolInvocation;

/// Scripted assistant step.
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// Model requests a tool call.
    ToolCall(ToolInvocation),
    /// Terminal message after tools (or when there are no tools).
    Done { message: String },
}

/// Event emitted while executing a plan (for assertions).
#[derive(Debug, Clone, PartialEq)]
pub enum RunEvent {
    ToolResult { tool_name: String, result: Value },
    Done { message: String },
    Aborted { error_json: Value },
}

/// Run a fixed plan: invoke tools in order; **abort** on first error (v1 policy).
pub async fn run_plan(
    registry: &ToolRegistry,
    steps: &[PlanStep],
) -> Result<Vec<RunEvent>, ToolInvokeError> {
    let mut events = Vec::new();
    for step in steps {
        match step {
            PlanStep::ToolCall(inv) => {
                match registry.invoke(&inv.tool_name, inv.arguments.clone()).await {
                    Ok(result) => events.push(RunEvent::ToolResult {
                        tool_name: inv.tool_name.clone(),
                        result,
                    }),
                    Err(e) => {
                        let j = e.to_json();
                        events.push(RunEvent::Aborted {
                            error_json: j.clone(),
                        });
                        return Err(e);
                    }
                }
            }
            PlanStep::Done { message } => {
                events.push(RunEvent::Done {
                    message: message.clone(),
                });
            }
        }
    }
    Ok(events)
}
