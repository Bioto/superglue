//! Per-turn guard against repeated identical file mutations.
//!
//! Observation tools always execute. Their results can change between calls.
//! File mutations may run once per identical argument set in a turn.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::file_locks::is_file_mutating_tool;

/// Tracks file-mutating tool invocations within a single agent turn.
#[derive(Debug, Default)]
pub struct ToolLoopGuard {
    seen_mutations: HashSet<(String, u64)>,
}

impl ToolLoopGuard {
    /// Returns a suppression payload when this exact file mutation already ran.
    ///
    /// Observation tools always return `None` so the caller executes them.
    pub fn check_suppressed(&mut self, tool_name: &str, args: &Value) -> Option<String> {
        if !is_file_mutating_tool(tool_name) {
            return None;
        }

        let key = call_key(tool_name, args);
        if !self.seen_mutations.insert(key) {
            tracing::info!(
                tool = tool_name,
                "tool loop guard suppressed identical file mutation"
            );
            return Some(suppressed_mutation_json());
        }
        None
    }
}

fn suppressed_mutation_json() -> String {
    json!({
        "ok": false,
        "suppressed": true,
        "reason": "identical_file_mutation",
        "message": "This exact file mutation already ran in this turn. Inspect current state before another edit."
    })
    .to_string()
}

pub type SharedToolLoopGuard = Arc<Mutex<ToolLoopGuard>>;

#[must_use]
pub fn new_tool_loop_guard() -> SharedToolLoopGuard {
    Arc::new(Mutex::new(ToolLoopGuard::default()))
}

fn call_key(tool_name: &str, args: &Value) -> (String, u64) {
    use std::collections::hash_map::DefaultHasher;
    let canonical = serde_json::to_string(args).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    (tool_name.to_string(), hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_after_edit_is_not_suppressed() {
        let mut guard = ToolLoopGuard::default();
        let read_args = json!({"path": "src/foo.rs"});
        let edit_args = json!({
            "path": "src/foo.rs",
            "old_string": "use path",
            "new_string": "use { join, path }",
        });

        assert!(guard.check_suppressed("read", &read_args).is_none());
        assert!(guard.check_suppressed("edit", &edit_args).is_none());
        assert!(
            guard.check_suppressed("read", &read_args).is_none(),
            "the same read must execute again after an edit"
        );
    }

    #[test]
    fn overlapping_read_is_not_intercepted() {
        let mut guard = ToolLoopGuard::default();
        let first = json!({
            "path": "src/foo.rs",
            "offset": 1588,
            "limit": 11
        });
        let shifted = json!({
            "path": "./src/foo.rs",
            "offset": 1578,
            "limit": 21
        });

        assert!(guard.check_suppressed("read", &first).is_none());
        assert!(
            guard.check_suppressed("read", &shifted).is_none(),
            "overlapping reads must execute so the model sees current bytes"
        );
    }

    #[test]
    fn second_identical_edit_returns_suppressed_not_cached_success() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "\"type\": \"array\",\n\"items\": {",
            "new_string": "\"type\": \"array\",\n\"minItems\": 1,\n\"items\": {",
        });

        assert!(guard.check_suppressed("edit", &args).is_none());
        let suppressed = guard
            .check_suppressed("edit", &args)
            .expect("identical edit should be suppressed");
        let parsed: Value = serde_json::from_str(&suppressed).expect("valid json");
        assert_eq!(parsed.get("ok").and_then(Value::as_bool), Some(false));
        assert_eq!(
            parsed.get("suppressed").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            parsed.get("reason").and_then(Value::as_str),
            Some("identical_file_mutation")
        );
        assert!(parsed.get("cached").is_none());
        assert!(parsed.get("rel_path").is_none());
        assert!(parsed.get("path").is_none());
        assert!(parsed.get("prior_result").is_none());
        assert!(!suppressed.contains("ok\":true"));
    }

    #[test]
    fn failed_edit_is_not_replayed_as_current_failure() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "missing anchor",
            "new_string": "replacement",
        });

        assert!(guard.check_suppressed("edit", &args).is_none());
        let suppressed = guard
            .check_suppressed("edit", &args)
            .expect("identical failing edit should be suppressed");
        assert!(
            !suppressed.contains("old_string not found"),
            "suppression must not copy a prior failure"
        );
        let parsed: Value = serde_json::from_str(&suppressed).expect("valid json");
        assert_eq!(
            parsed.get("reason").and_then(Value::as_str),
            Some("identical_file_mutation")
        );
    }

    #[test]
    fn different_edits_remain_executable() {
        let mut guard = ToolLoopGuard::default();
        let base = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "\"type\": \"array\",\n\"items\": {",
            "new_string": "questions variant",
        });
        let sibling = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "\"type\": \"array\",\n\"items\": {",
            "new_string": "options variant",
        });
        assert!(guard.check_suppressed("edit", &base).is_none());
        assert!(guard.check_suppressed("edit", &sibling).is_none());
    }

    #[test]
    fn observation_tools_always_execute() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({"path": "src/foo.rs", "pattern": "join"});
        assert!(guard.check_suppressed("grep", &args).is_none());
        assert!(guard.check_suppressed("grep", &args).is_none());
        assert!(guard.check_suppressed("glob", &args).is_none());
        assert!(guard.check_suppressed("glob", &args).is_none());
        assert!(
            guard
                .check_suppressed("shell", &json!({"command": "ls"}))
                .is_none()
        );
        assert!(
            guard
                .check_suppressed("shell", &json!({"command": "ls"}))
                .is_none()
        );
    }

    #[test]
    fn write_and_apply_patch_aliases_are_suppressed() {
        let mut guard = ToolLoopGuard::default();
        let write_args = json!({"path": "src/foo.rs", "content": "hello"});
        assert!(guard.check_suppressed("write", &write_args).is_none());
        assert!(guard.check_suppressed("write", &write_args).is_some());
        assert!(guard.check_suppressed("write_file", &write_args).is_none());
        assert!(guard.check_suppressed("write_file", &write_args).is_some());

        let patch_args = json!({"path": "src/foo.rs", "patch": "@@"});
        assert!(guard.check_suppressed("apply_patch", &patch_args).is_none());
        assert!(guard.check_suppressed("apply_patch", &patch_args).is_some());
    }
}
