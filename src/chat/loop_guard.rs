//! Per-turn guard against identical tool-call loops.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde_json::{Value, json};
use tokio::sync::Mutex;

const MAX_IDENTICAL_CALLS: u32 = 2;

/// Tracks recent identical tool invocations within a single agent turn.
#[derive(Debug, Default)]
pub struct ToolLoopGuard {
    calls: HashMap<(String, u64), CallRecord>,
}

#[derive(Debug, Default)]
struct CallRecord {
    count: u32,
    last_result: Option<String>,
}

impl ToolLoopGuard {
    /// Returns a cached JSON result when the same tool+args has already run twice.
    pub fn check_cached(&mut self, tool_name: &str, args: &Value) -> Option<String> {
        let key = call_key(tool_name, args);
        let record = self.calls.entry(key).or_default();
        record.count += 1;
        if record.count > MAX_IDENTICAL_CALLS {
            return record.last_result.as_ref().map(|prior| {
                let note = format!(
                    "Identical `{tool_name}` call repeated {count} times; returning prior result. \
                     Do not call again with the same arguments — use the cached output or change approach.",
                    count = record.count
                );
                cached_result_json(prior, &note)
            });
        }
        None
    }

    pub fn record(&mut self, tool_name: &str, args: &Value, result_json: &str) {
        let key = call_key(tool_name, args);
        if let Some(record) = self.calls.get_mut(&key) {
            record.last_result = Some(result_json.to_string());
        }
    }
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
    fn third_identical_call_returns_cache() {
        let mut guard = ToolLoopGuard::default();
        let args = json!({"path": "src/foo.rs"});
        assert!(guard.check_cached("read", &args).is_none());
        guard.record("read", &args, r#"{"content":"hello","path":"src/foo.rs"}"#);
        assert!(guard.check_cached("read", &args).is_none());
        guard.record("read", &args, r#"{"content":"hello","path":"src/foo.rs"}"#);
        let cached = guard.check_cached("read", &args).expect("cached on third");
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
        assert!(guard.check_cached("echo", &args).is_none());
        guard.record("echo", &args, r#""plain string result""#);
        let cached = guard.check_cached("echo", &args).expect("cached on third");
        let parsed: Value = serde_json::from_str(&cached).expect("valid json");
        assert_eq!(parsed.get("cached").and_then(|c| c.as_bool()), Some(true));
        assert!(parsed.get("prior_result").is_some());
    }
}
