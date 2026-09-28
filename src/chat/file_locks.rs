//! Per-path serialization for file-mutating tool calls.
//!
//! Tool calls within a round dispatch concurrently, so two `edit` calls targeting
//! the same file race: the first rewrites the file and the second fails with
//! `old_string not found`. Mutating tools take a per-path lock so same-file calls
//! run sequentially while different files stay parallel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde_json::Value;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Tools that rewrite a file in place and therefore must not run concurrently
/// against the same path.
const FILE_MUTATING_TOOLS: &[&str] = &[
    "edit",
    "edit_file",
    "write",
    "write_file",
    "patch",
    "apply_patch",
    "multi_edit",
];

type LockMap = HashMap<String, Weak<AsyncMutex<()>>>;

fn locks() -> &'static Mutex<LockMap> {
    static LOCKS: OnceLock<Mutex<LockMap>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Acquire the write lock for `args.path` when `tool_name` mutates a file.
///
/// Returns `None` when the tool is not file-mutating or no path is present, in
/// which case the caller proceeds without serialization. The returned guard must
/// be held for the duration of the tool invocation.
pub(crate) async fn acquire_file_lock(
    tool_name: &str,
    args: &Value,
) -> Option<OwnedMutexGuard<()>> {
    if !is_file_mutating_tool(tool_name) {
        return None;
    }
    let path = path_from_args(args)?;

    let lock = {
        let mut map = locks().lock().ok()?;
        // Entries are weak so finished paths drop out; sweep opportunistically.
        if map.len() > 256 {
            map.retain(|_, weak| weak.strong_count() > 0);
        }
        match map.get(&path).and_then(Weak::upgrade) {
            Some(existing) => existing,
            None => {
                let fresh = Arc::new(AsyncMutex::new(()));
                map.insert(path, Arc::downgrade(&fresh));
                fresh
            }
        }
    };

    Some(lock.lock_owned().await)
}

pub(crate) fn is_file_mutating_tool(tool_name: &str) -> bool {
    FILE_MUTATING_TOOLS.contains(&tool_name)
}

fn path_from_args(args: &Value) -> Option<String> {
    let raw = args
        .get("path")
        .or_else(|| args.get("file_path"))
        .and_then(Value::as_str)?
        .trim();
    if raw.is_empty() {
        return None;
    }
    Some(normalize_path(raw))
}

/// Collapse `./` prefixes and trailing separators so the same file maps to one key.
fn normalize_path(path: &str) -> String {
    let trimmed = path.trim().trim_end_matches('/');
    trimmed
        .strip_prefix("./")
        .unwrap_or(trimmed)
        .replace("//", "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn only_mutating_tools_lock() {
        assert!(is_file_mutating_tool("edit"));
        assert!(is_file_mutating_tool("write_file"));
        assert!(!is_file_mutating_tool("read"));
        assert!(!is_file_mutating_tool("grep"));
    }

    #[test]
    fn path_variants_normalize_to_one_key() {
        assert_eq!(
            path_from_args(&json!({"path": "./src/main.rs"})),
            Some("src/main.rs".to_string())
        );
        assert_eq!(
            path_from_args(&json!({"file_path": "src/main.rs"})),
            Some("src/main.rs".to_string())
        );
        assert_eq!(path_from_args(&json!({"path": ""})), None);
        assert_eq!(path_from_args(&json!({})), None);
    }

    #[tokio::test]
    async fn same_path_edits_serialize() {
        let counter = Arc::new(AtomicU32::new(0));
        let overlaps = Arc::new(AtomicU32::new(0));
        let args = json!({"path": "src/same.rs"});

        let tasks: Vec<_> = (0..8)
            .map(|_| {
                let counter = Arc::clone(&counter);
                let overlaps = Arc::clone(&overlaps);
                let args = args.clone();
                tokio::spawn(async move {
                    let _guard = acquire_file_lock("edit", &args).await;
                    if counter.fetch_add(1, Ordering::SeqCst) != 0 {
                        overlaps.fetch_add(1, Ordering::SeqCst);
                    }
                    tokio::task::yield_now().await;
                    counter.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(overlaps.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn different_paths_do_not_block_each_other() {
        let first = acquire_file_lock("edit", &json!({"path": "a.rs"})).await;
        let second = acquire_file_lock("edit", &json!({"path": "b.rs"})).await;
        assert!(first.is_some());
        assert!(second.is_some());
    }

    #[tokio::test]
    async fn read_tool_is_never_locked() {
        let guard = acquire_file_lock("read", &json!({"path": "a.rs"})).await;
        assert!(guard.is_none());
    }
}
