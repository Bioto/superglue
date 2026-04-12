//! JSON tool boundary: registration, async dispatch, and test harness.
//!
//! See [`crate`] docs and `_docs/03-tool-boundary-and-async.md`.

mod error;
mod harness;
mod registry;
pub mod types;

pub use error::ToolInvokeError;
pub use harness::{PlanStep, RunEvent, run_plan};
pub use registry::{Tool, ToolRegistry};
pub use types::{ToolInvocation, ToolSpec};
