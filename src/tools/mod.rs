//! JSON tool boundary: registration, async dispatch, and test harness.
//!
//! See [`crate`] docs and `_docs/03-tool-boundary-and-async.md`.

mod error;
mod harness;
pub mod router;
mod registry;
pub mod types;

pub use error::ToolInvokeError;
pub use harness::{PlanStep, RunEvent, run_plan};
pub use registry::{Tool, ToolRegistry};
pub use router::{
    ActiveToolSet, ToolMode, ROUTER_TOOL_NAME, build_router_tool_spec, is_router_call,
    router_query_from_calls, DEFAULT_TOOL_ROUTE_MODEL,
};
pub use types::{OnToolError, ToolInvocation, ToolRetryPolicy, ToolSpec};
