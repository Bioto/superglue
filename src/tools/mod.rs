//! JSON tool boundary: registration, async dispatch, and test harness.
//!
//! See [`crate`] docs and `_docs/03-tool-boundary-and-async.md`.

pub mod code;
mod error;
mod harness;
mod registry;
pub mod router;
pub mod types;

pub use code::{CodeLimits, execute_code, source_from_arguments};
pub use error::ToolInvokeError;
pub use harness::{PlanStep, RunEvent, run_plan};
pub use registry::{Tool, ToolRegistry};
pub use router::{
    ActiveToolSet, CODE_TOOL_NAME, DEFAULT_TOOL_ROUTE_MODEL, ROUTER_TOOL_NAME, ToolMode,
    build_code_tool_spec, build_router_tool_spec, is_code_call, is_router_call,
    router_call_id_from_calls, router_query_from_calls,
};
pub use types::{OnToolError, ToolContextPolicy, ToolInvocation, ToolRetryPolicy, ToolSpec};
