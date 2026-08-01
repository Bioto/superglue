//! Per-turn guard against identical and overlapping tool-call loops.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::Mutex;

const MAX_IDENTICAL_CALLS: u32 = 1;
const MAX_READ_COVERAGE_RECORDS: usize = 128;

/// Tracks repeated tool invocations within a single agent turn.
#[derive(Debug, Default)]
pub struct ToolLoopGuard {
    calls: HashMap<(String, u64), CallRecord>,
    read_calls: HashMap<String, Vec<ReadCallRecord>>,
}

#[derive(Debug, Default)]
struct CallRecord {
    count: u32,
    last_result: Option<String>,
}

#[derive(Debug)]
struct ReadCallRecord {
    start_line: u64,
    end_line: u64,
    last_result: Option<String>,
}

impl ToolLoopGuard {
    /// Returns a cached JSON result when a tool call is an exact repeat or a
    /// substantial overlap of a previously completed file read.
    pub fn check_cached(&mut self, tool_name: &str, args: &Value) -> Option<String> {
        if tool_name == "read" {
            let ranges = read_ranges(args);
            for range in ranges {
                if let Some(prior) = self.read_calls.get(&range.path).and_then(|calls| {
                    calls.iter().find(|call| {
                        ranges_overlap(
                            range.start_line,
                            range.end_line,
                            call.start_line,
                            call.end_line,
                        )
                    })
                }) && let Some(result) = &prior.last_result
                && !is_notepad_stub_result(result)
                {
                    let note = format!(
                        "Read of `{}` lines {}-{} substantially overlaps an earlier read; \
                         returning the cached result. Use the cached content or read a \
                         disjoint range instead.",
                        range.path, range.start_line, range.end_line
                    );
                    tracing::info!(
                        tool = tool_name,
                        path = %range.path,
                        start_line = range.start_line,
                        end_line = range.end_line,
                        "tool loop guard returned cached overlapping read"
                    );
                    return Some(cached_result_json(result, &note));
                }
            }
        }

        let key = call_key(tool_name, args);
        let record = self.calls.entry(key).or_default();
        record.count += 1;
        if record.count >= MAX_IDENTICAL_CALLS
            && let Some(prior) = record.last_result.as_ref()
            && !is_notepad_stub_result(prior)
        {
            let note = format!(
                "Identical `{tool_name}` call repeated {count} times; returning prior result. \
                 Do not call again with the same arguments — use the cached output or change approach.",
                count = record.count
            );
            tracing::info!(
                tool = tool_name,
                repeat_count = record.count,
                "tool loop guard returned cached identical call"
            );
            return Some(cached_result_json(prior, &note));
        }
        None
    }

    pub fn record(&mut self, tool_name: &str, args: &Value, result_json: &str) {
        if is_notepad_stub_result(result_json) {
            return;
        }
        if tool_name == "read" {
            for range in read_ranges(args) {
                let calls = self.read_calls.entry(range.path).or_default();
                if let Some(existing) = calls.iter_mut().find(|call| {
                    call.start_line == range.start_line && call.end_line == range.end_line
                }) {
                    existing.last_result = Some(result_json.to_string());
                } else {
                    calls.push(ReadCallRecord {
                        start_line: range.start_line,
                        end_line: range.end_line,
                        last_result: Some(result_json.to_string()),
                    });
                    if calls.len() > MAX_READ_COVERAGE_RECORDS {
                        calls.remove(0);
                    }
                }
            }
        }
        let key = call_key(tool_name, args);
        if let Some(record) = self.calls.get_mut(&key) {
            record.last_result = Some(result_json.to_string());
        }
    }
}

#[derive(Debug)]
struct ReadRange {
    path: String,
    start_line: u64,
    end_line: u64,
}

fn read_ranges(args: &Value) -> Vec<ReadRange> {
    let Some(object) = args.as_object() else {
        return Vec::new();
    };
    let shared_offset = object.get("offset").and_then(Value::as_u64).unwrap_or(1);
    let shared_limit = object.get("limit").and_then(Value::as_u64).unwrap_or(200);
    let range_for = |path: &str, offset: u64, limit: u64| ReadRange {
        path: normalize_path(path),
        start_line: offset.max(1),
        end_line: offset
            .max(1)
            .saturating_add(limit.clamp(1, 500))
            .saturating_sub(1),
    };

    if let Some(files) = object.get("files").and_then(Value::as_array) {
        return files
            .iter()
            .filter_map(|file| {
                let file = file.as_object()?;
                let path = file.get("path").and_then(Value::as_str)?;
                Some(range_for(
                    path,
                    file.get("offset")
                        .and_then(Value::as_u64)
                        .unwrap_or(shared_offset),
                    file.get("limit")
                        .and_then(Value::as_u64)
                        .unwrap_or(shared_limit),
                ))
            })
            .collect();
    }

    if let Some(paths) = object.get("paths").and_then(Value::as_array) {
        return paths
            .iter()
            .filter_map(Value::as_str)
            .map(|path| range_for(path, shared_offset, shared_limit))
            .collect();
    }

    object
        .get("path")
        .and_then(Value::as_str)
        .map(|path| vec![range_for(path, shared_offset, shared_limit)])
        .unwrap_or_default()
}

fn normalize_path(path: &str) -> String {
    path.trim()
        .strip_prefix("./")
        .unwrap_or(path.trim())
        .to_string()
}

fn ranges_overlap(start: u64, end: u64, prior_start: u64, prior_end: u64) -> bool {
    let overlap_start = start.max(prior_start);
    let overlap_end = end.min(prior_end);
    if overlap_start > overlap_end {
        return false;
    }
    let overlap = overlap_end - overlap_start + 1;
    let requested = end.saturating_sub(start) + 1;
    overlap.saturating_mul(2) >= requested
}

/// Merge cache metadata into the prior result when it is a JSON object so
/// identity fields (`path`, `rel_path`, `match_count`, …) stay at the top level.
fn cached_result_json(prior: &str, note: &str) -> String {
    match serde_json::from_str::<Value>(prior) {
        Ok(Value::Object(mut map)) => {
            map.insert("cached".into(), json!(true));
            map.insert("note".into(), json!(note));
            Value::Object(map).to_string()
        }
        Ok(other) => json!({
            "ok": true,
            "cached": true,
            "note": note,
            "prior_result": other,
        })
        .to_string(),
        Err(_) => json!({
            "ok": true,
            "cached": true,
            "note": note,
            "prior_result": prior,
        })
        .to_string(),
    }
}

pub type SharedToolLoopGuard = Arc<Mutex<ToolLoopGuard>>;

#[must_use]
pub fn new_tool_loop_guard() -> SharedToolLoopGuard {
    Arc::new(Mutex::new(ToolLoopGuard::default()))
}

fn is_notepad_stub_result(result_json: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(result_json) else {
        return false;
    };
    if value.get("notepad_ref").and_then(Value::as_str).is_some() {
        return true;
    }
    value
        .get("notepad_refs")
        .and_then(Value::as_array)
        .is_some_and(|arr| !arr.is_empty())
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
    fn repeated_identical_edit_returns_cache_instead_of_reapplying() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "\"type\": \"array\",\n\"items\": {",
            "new_string": "\"type\": \"array\",\n\"minItems\": 1,\n\"items\": {",
        });
        assert!(guard.check_cached("edit", &args).is_none());
        guard.record(
            "edit",
            &args,
            r#"{"ok":true,"rel_path":"src/tools/prompt.rs"}"#,
        );

        // Re-issuing the same edit later in the turn must not run against the
        // already-rewritten file.
        let cached = guard
            .check_cached("edit", &args)
            .expect("identical edit should be cached");
        assert!(cached.contains("\"cached\":true"));
    }

    #[test]
    fn repeated_failing_edit_returns_cached_failure() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({
            "path": "src/tools/prompt.rs",
            "old_string": "missing anchor",
            "new_string": "replacement",
        });
        assert!(guard.check_cached("edit", &args).is_none());
        // Skip policy surfaces tool failures as Ok(..) payloads, so they record.
        guard.record(
            "edit",
            &args,
            r#"{"ok":false,"error":"old_string not found"}"#,
        );

        let cached = guard
            .check_cached("edit", &args)
            .expect("identical failing edit should be cached");
        assert!(cached.contains("old_string not found"));
        assert!(cached.contains("\"cached\":true"));
    }

    #[test]
    fn edits_differing_only_in_new_string_are_not_cached() {
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
        assert!(guard.check_cached("edit", &base).is_none());
        guard.record("edit", &base, r#"{"ok":true}"#);
        assert!(guard.check_cached("edit", &sibling).is_none());
    }

    #[test]
    fn second_identical_call_returns_cache() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({"path": "src/foo.rs"});
        assert!(guard.check_cached("read", &args).is_none());
        guard.record("read", &args, r#"{"content":"hello","path":"src/foo.rs"}"#);
        let cached = guard.check_cached("read", &args).expect("cached on second");
        assert!(cached.contains("\"cached\":true"));
        assert!(cached.contains("hello"));
        let parsed: Value = serde_json::from_str(&cached).expect("valid json");
        assert_eq!(
            parsed.get("path").and_then(|p| p.as_str()),
            Some("src/foo.rs")
        );
        assert_eq!(
            parsed.get("content").and_then(|c| c.as_str()),
            Some("hello")
        );
        assert_eq!(parsed.get("cached").and_then(|c| c.as_bool()), Some(true));
        assert!(parsed.get("note").and_then(|n| n.as_str()).is_some());
        assert!(parsed.get("prior_result").is_none());
    }

    #[test]
    fn cached_non_object_prior_nests_under_prior_result() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({"q": "x"});
        assert!(guard.check_cached("echo", &args).is_none());
        guard.record("echo", &args, r#""plain string result""#);
        let cached = guard.check_cached("echo", &args).expect("cached on second");
        let parsed: Value = serde_json::from_str(&cached).expect("valid json");
        assert_eq!(parsed.get("cached").and_then(|c| c.as_bool()), Some(true));
        assert!(parsed.get("prior_result").is_some());
    }

    #[test]
    fn overlapping_read_ranges_return_cached_result() {
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
        assert!(guard.check_cached("read", &first).is_none());
        guard.record(
            "read",
            &first,
            r#"{"path":"src/foo.rs","start_line":1588,"end_line":1598,"content":"exact bytes"}"#,
        );

        let cached = guard
            .check_cached("read", &shifted)
            .expect("overlapping read should be cached");
        assert!(cached.contains("\"cached\":true"));
        assert!(cached.contains("exact bytes"));
        assert!(cached.contains("substantially overlaps"));
    }

    #[test]
    fn notepad_stub_reads_are_not_cached_or_recorded() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({"path": "src/foo.rs", "offset": 1, "limit": 200});
        let stub = json!({
            "notepad_ref": "abc-123",
            "stored_chars": 25000,
            "preview": "  1| use std::path::Path;",
            "retrieve": {"tool": "notepad", "action": "read", "id": "abc-123"},
        })
        .to_string();
        assert!(guard.check_cached("read", &args).is_none());
        guard.record("read", &args, &stub);
        assert!(
            guard.check_cached("read", &args).is_none(),
            "stub must not be served as cached content"
        );
    }

    #[test]
    fn disjoint_read_ranges_are_not_cached() {
        let mut guard = ToolLoopGuard::default();
        let first = json!({"path": "src/foo.rs", "offset": 10, "limit": 10});
        let disjoint = json!({"path": "src/foo.rs", "offset": 30, "limit": 10});
        assert!(guard.check_cached("read", &first).is_none());
        guard.record("read", &first, r#"{"content":"first"}"#);
        assert!(guard.check_cached("read", &disjoint).is_none());
    }
}
