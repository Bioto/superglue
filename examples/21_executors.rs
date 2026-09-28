//! Example 21 — Executor pattern (Python only).
//!
//! SimpleExecutor, AgentExecutor, and AgentStructuredExecutor are not exposed in Rust.
//! See `superglue-py/examples/21_executors.py`.

fn main() {
    println!(
        "Executor types (SimpleExecutor, AgentExecutor, AgentStructuredExecutor) \
         are not exposed in the Rust crate.\n\
         Use Client, AgentEngine, hooks, and guardrails directly, or run \
         superglue-py/examples/21_executors.py in Python.\n\n\
         See examples/README.md."
    );
}
