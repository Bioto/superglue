//! Short UI summaries for tool results, computed from the full payload before truncation.

use serde_json::Value;

/// One-line summary for tool status events (TUI / observability).
pub fn tool_result_summary(tool_name: &str, content: &str) -> Option<String> {
    let v = serde_json::from_str::<Value>(content).ok()?;
    if v.get("ok") == Some(&Value::Bool(false)) {
        return v
            .get("error")
            .and_then(|e| e.as_str())
            .map(|e| format!("error: {}", truncate_chars(e, 100)))
            .or(Some("error".into()));
    }
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Some(truncate_chars(err, 100));
    }
    Some(match tool_name {
        "list_dir" => {
            let n = v
                .get("entries")
                .and_then(|e| e.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let truncated = v
                .get("truncated")
                .and_then(|t| t.as_bool())
                .unwrap_or(false);
            let mut out = if n == 1 {
                "Scanned 1 entry".into()
            } else {
                format!("Scanned {n} entries")
            };
            if truncated {
                out.push_str(" (truncated)");
            }
            out
        }
        "glob" => {
            let n = v.get("match_count").and_then(|c| c.as_u64()).unwrap_or(0);
            format_found(n, "file")
        }
        "grep" => {
            let n = v.get("match_count").and_then(|c| c.as_u64()).unwrap_or(0);
            let unit =
                if v.get("output_mode").and_then(|m| m.as_str()) == Some("files_with_matches") {
                    "file"
                } else {
                    "match"
                };
            format_found(n, unit)
        }
        "read_file" => {
            let path = v.get("path").and_then(|p| p.as_str()).unwrap_or("?");
            if let (Some(start), Some(end), Some(total)) = (
                v.get("start_line").and_then(|l| l.as_u64()),
                v.get("end_line").and_then(|l| l.as_u64()),
                v.get("total_lines").and_then(|l| l.as_u64()),
            ) {
                format!("Read lines {start}-{end} of {path} ({total} total)")
            } else {
                format!("Read {path}")
            }
        }
        "write_file" | "edit_file" | "apply_patch" => {
            let path = v
                .get("rel_path")
                .or_else(|| v.get("path"))
                .and_then(|p| p.as_str())
                .unwrap_or("?");
            if v.get("bytes_written").is_some() {
                format!("Wrote {path}")
            } else {
                format!("Edited {path}")
            }
        }
        "run_command" => {
            let code = v
                .get("exit_code")
                .and_then(|c| c.as_i64())
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into());
            format!("Exit {code}")
        }
        "search_code" => {
            let n = v
                .get("result_count")
                .and_then(|c| c.as_u64())
                .or_else(|| {
                    v.get("results")
                        .and_then(|r| r.as_array())
                        .map(|a| a.len() as u64)
                })
                .unwrap_or(0);
            format!("Found {n} results")
        }
        _ => return None,
    })
}

fn format_found(n: u64, unit: &str) -> String {
    if n == 1 {
        format!("Found 1 {unit}")
    } else {
        format!("Found {n} {unit}s")
    }
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    format!("{}…", text.chars().take(max).collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_dir_summary() {
        let raw = r#"{"path":".","entries":[{"kind":"dir","path":"src"}],"truncated":false}"#;
        assert_eq!(
            tool_result_summary("list_dir", raw).as_deref(),
            Some("Scanned 1 entry")
        );
    }
}
