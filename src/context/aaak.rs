//! AAAK lossless shorthand encoding (GlueLLM parity).

use serde_json::Value;

use crate::openai::{ChatMessage, MessageContent, ToolCall};

pub(crate) const AAAK_PREAMBLE_MARKER: &str = "[AAAK decoding hint]";

const AAAK_DECODE_HINT: &str = "Decode [AAAK CTX] and [AT]: USR:/AST: turns; T:name()→result for tools (args only when same name used twice).\n\
Lists [a,b,c]; ordered steps 1→2→3; config key=val; schema col:purpose; \
security attrs verbatim (HttpOnly,Secure,SameSite=Strict).";

pub(crate) const COMPRESS_SYSTEM: &str = "You are an expert lossless compressor. Convert the transcript into AAAK shorthand. \
You MUST recover every technical fact exactly — especially rate-limit layers, \
schema column purposes, cookie flags, and numbered steps.\n\n\
AAAK lossless shorthand — rules:\n\
ROLES: USR: AST: T:name()→val | T:name(key=val)→val (args only when disambiguating)\n\
LISTS: items in [] brackets: [a,b,c] | ordered steps: 1→2→3→4\n\
ATTRS: keep security/config attrs verbatim: HttpOnly,Secure,SameSite=Strict\n\
SCHEMA: col:purpose notation: replaced_by:replay_detect\n\
CONFIG: key=val pairs: timeout_ms=8500 | pool_size=10\n\
NUMBERS: preserve exact values, units: 847errors | 15min | 8500ms\n\
FACTS: separate with | ; chain with → ; relate with .\n\
BREVITY: 3-4 char handles for names only, never for config keys or values\n\n\
When you see [AT] blocks, first decode them (T:name()→val) then re-encode \
the extracted facts using the same rules. Output ONLY the AAAK-encoded lines \
(no markdown fences, no preamble). Every fact from the transcript must be \
recoverable from your encoding. No explanations.";

pub(crate) const COMPRESS_USER_PREFIX: &str =
    "Encode in AAAK. MUST preserve ALL technical facts exactly. Output ONLY AAAK lines.\n\n";

pub struct AaakCompressor;

impl AaakCompressor {
    #[must_use]
    pub fn get_spec_preamble() -> String {
        format!("{AAAK_PREAMBLE_MARKER}\n{AAAK_DECODE_HINT}")
    }

    pub fn ensure_preamble_in_system(message: &mut ChatMessage) {
        let Some(MessageContent::Text(content)) = message.content.as_mut() else {
            return;
        };
        if content.contains(AAAK_PREAMBLE_MARKER) {
            return;
        }
        let trimmed = content.trim_end();
        *content = format!("{trimmed}\n\n{}", Self::get_spec_preamble());
    }

    /// Deterministic `[AT]` block for one tool round (no LLM call).
    #[must_use]
    pub fn encode_tool_round(
        tool_calls: &[ToolCall],
        tool_messages: &[ChatMessage],
        id_to_name: &std::collections::HashMap<String, String>,
    ) -> String {
        let mut args_by_id = std::collections::HashMap::new();
        for tc in tool_calls {
            args_by_id.insert(tc.id.clone(), tc.function.arguments.clone());
        }

        let mut name_counts = std::collections::HashMap::<&str, usize>::new();
        for name in id_to_name.values() {
            *name_counts.entry(name.as_str()).or_insert(0) += 1;
        }

        let mut out = String::from("[AT]\n");
        for (idx, tool_msg) in tool_messages.iter().enumerate() {
            if idx > 0 {
                out.push_str(" | ");
            }
            let tc_id = tool_msg.tool_call_id.as_deref().unwrap_or("");
            let name = id_to_name.get(tc_id).map(String::as_str).unwrap_or(tc_id);
            let args = args_by_id.get(tc_id).map(String::as_str).unwrap_or("{}");
            let raw = tool_msg
                .content
                .as_ref()
                .and_then(|c| c.as_text())
                .unwrap_or("");
            let formatted_args = if name_counts.get(name).copied().unwrap_or(0) > 1 {
                format_tool_args(args)
            } else {
                String::new()
            };
            out.push_str("T:");
            out.push_str(name);
            out.push('(');
            out.push_str(&formatted_args);
            out.push_str(")→");
            out.push_str(&format_tool_result(raw, 2000));
        }
        out
    }
}

pub(crate) fn message_text(msg: &ChatMessage) -> String {
    msg.content
        .as_ref()
        .and_then(|c| c.as_text())
        .unwrap_or("")
        .to_string()
}

pub(crate) fn extract_aaak_ctx_inner(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if !trimmed.contains("[AAAK CTX]") {
        return None;
    }
    let without_open = trimmed.split("[AAAK CTX]").nth(1).unwrap_or(trimmed);
    let inner = without_open
        .split("[/AAAK CTX]")
        .next()
        .unwrap_or(without_open)
        .trim();
    if inner.is_empty() {
        None
    } else {
        Some(inner.to_string())
    }
}

/// Merge already-compressed AAAK context with newer turns without another LLM call.
pub(crate) fn passthrough_aaak_context(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let text = message_text(msg).trim().to_string();
        if text.contains("[AAAK CTX]") {
            if let Some(inner) = extract_aaak_ctx_inner(&text) {
                parts.push(inner);
            }
            continue;
        }
        if text.starts_with("[AT]") {
            parts.push(text);
            continue;
        }
        match msg.role.as_str() {
            "user" => parts.push(format!("USR: {text}")),
            "assistant" => parts.push(format!("AST: {text}")),
            _ => parts.push(text),
        }
    }
    parts.join("\n")
}

/// When condensing replaces a tool round, tell the model not to re-invoke the same tools.
pub(crate) const CONDENSE_ANTI_LOOP_SUFFIX: &str = "\n(completed tool results for this round — do not re-invoke these same tool calls; \
     you may call new tools if still needed to finish the user request)";

pub(crate) fn passthrough_at_messages(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let content = message_text(msg).trim().to_string();
        match msg.role.as_str() {
            "assistant" if content.starts_with("[AT]") => parts.push(content),
            "user" => parts.push(format!("USR: {content}")),
            "assistant" => parts.push(format!("AST: {content}")),
            "tool" => {
                let tc_id = msg.tool_call_id.as_deref().unwrap_or("");
                parts.push(format!("TOOL_RESULT[{tc_id}]: {content}"));
            }
            role => parts.push(format!("{}: {content}", role.to_uppercase())),
        }
    }
    parts.join("\n")
}

pub(crate) fn transcript_from_messages(messages: &[ChatMessage]) -> String {
    let mut lines = Vec::new();
    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                let tc_id = msg.tool_call_id.as_deref().unwrap_or("");
                let body = message_text(msg);
                lines.push(format!("TOOL_RESULT[{tc_id}]: {body}"));
            }
            "assistant" if msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) => {
                let mut tc_parts = Vec::new();
                for tc in msg.tool_calls.as_ref().unwrap_or(&vec![]) {
                    tc_parts.push(format!("{}({})", tc.function.name, tc.function.arguments));
                }
                let content = message_text(msg);
                let suffix = if tc_parts.is_empty() {
                    String::new()
                } else {
                    format!(" | calls: {}", tc_parts.join(", "))
                };
                lines.push(format!("ASSISTANT:{suffix}\n{content}").trim().to_string());
            }
            _ => {
                let content = message_text(msg);
                lines.push(format!("{}: {content}", msg.role.to_uppercase()));
            }
        }
    }
    lines.join("\n")
}

fn escape_aaak_value(value: &str, max_len: usize) -> String {
    let s = value
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('|', "\\|");
    if s.chars().count() > max_len {
        format!("{}...", s.chars().take(max_len - 3).collect::<String>())
    } else {
        s
    }
}

fn format_scalar_for_flatten(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                let s = format!("{f:.15}");
                s.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                n.to_string()
            }
        }
        Value::String(s) => {
            let escaped = s
                .replace('\\', "\\\\")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('|', "\\|");
            if s.is_empty() || s.contains([' ', '\t', '"', '=']) {
                format!("\"{}\"", escaped.replace('"', "\\\""))
            } else {
                escaped
            }
        }
        _ => format_scalar_for_flatten(&Value::String(v.to_string())),
    }
}

fn flatten_dict_lines(d: &serde_json::Map<String, Value>, prefix: &str) -> Vec<String> {
    if d.is_empty() {
        return vec![if prefix.is_empty() {
            "<empty>".into()
        } else {
            format!("{prefix}=<empty>")
        }];
    }
    let mut keys: Vec<_> = d.keys().collect();
    keys.sort();
    let mut lines = Vec::new();
    for k in keys {
        let v = &d[k];
        let p = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}")
        };
        match v {
            Value::Object(nested) => lines.extend(flatten_dict_lines(nested, &p)),
            Value::Array(arr) => lines.extend(flatten_list_lines(arr, &p)),
            _ => lines.push(format!("{p}={}", format_scalar_for_flatten(v))),
        }
    }
    lines
}

fn flatten_list_lines(arr: &[Value], prefix: &str) -> Vec<String> {
    if arr.is_empty() {
        return vec![if prefix.is_empty() {
            "<empty>".into()
        } else {
            format!("{prefix}=[]")
        }];
    }
    let mut lines = Vec::new();
    for (i, item) in arr.iter().enumerate() {
        let label = if prefix.is_empty() {
            format!("[{i}]")
        } else {
            format!("{prefix}[{i}]")
        };
        match item {
            Value::Object(map) => {
                let mut scalar_parts = Vec::new();
                let mut nested = Vec::new();
                let mut keys: Vec<_> = map.keys().collect();
                keys.sort();
                for k in keys {
                    let v = &map[k];
                    let sub = format!("{label}.{k}");
                    match v {
                        Value::Object(o) => nested.extend(flatten_dict_lines(o, &sub)),
                        Value::Array(a) => nested.extend(flatten_list_lines(a, &sub)),
                        _ => scalar_parts.push(format!("{k}={}", format_scalar_for_flatten(v))),
                    }
                }
                if !scalar_parts.is_empty() {
                    lines.push(format!("{label} {}", scalar_parts.join(" ")));
                } else if nested.is_empty() {
                    lines.push(format!("{label} <empty>"));
                }
                lines.extend(nested);
            }
            Value::Array(a) => lines.extend(flatten_list_lines(a, &label)),
            _ => lines.push(format!("{label}={}", format_scalar_for_flatten(item))),
        }
    }
    lines
}

fn flatten_json_to_body(obj: &Value) -> String {
    let inner = match obj {
        Value::Object(d) => flatten_dict_lines(d, ""),
        Value::Array(a) => flatten_list_lines(a, ""),
        other => vec![format_scalar_for_flatten(other)],
    };
    format!(
        "\n{}",
        inner
            .iter()
            .map(|ln| format!("  {ln}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn format_tool_result(raw: &str, max_len: usize) -> String {
    let s = raw.trim();
    if s.starts_with('{') || s.starts_with('[') {
        if let Ok(obj) = serde_json::from_str::<Value>(s) {
            let flat = flatten_json_to_body(&obj);
            if flat.chars().count() > max_len {
                return format!("{}...", flat.chars().take(max_len - 3).collect::<String>());
            }
            return flat;
        }
    }
    if s.contains('\n') {
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() > 1 {
            let header = lines[0].trim();
            if header.contains(',') && !header.contains(':') {
                let comma_count = header.matches(',').count();
                let data_rows: Vec<&str> = lines[1..]
                    .iter()
                    .copied()
                    .filter(|ln| ln.trim().matches(',').count() == comma_count)
                    .collect();
                if data_rows.len() >= 2 {
                    let stats = csv_stats_comment(header.split(',').collect(), &data_rows);
                    let annotated = if stats.is_empty() {
                        s.to_string()
                    } else {
                        format!("{s}\n{stats}")
                    };
                    if annotated.chars().count() > max_len {
                        return format!(
                            "{}...",
                            annotated.chars().take(max_len - 3).collect::<String>()
                        );
                    }
                    return annotated;
                }
            }
        }
        let body = s
            .lines()
            .map(|ln| format!("  {}", ln.replace('|', "\\|")))
            .collect::<Vec<_>>()
            .join("\n");
        let preserved = format!("\n{body}");
        if preserved.chars().count() > max_len {
            return format!(
                "{}...",
                preserved.chars().take(max_len - 3).collect::<String>()
            );
        }
        return preserved;
    }
    escape_aaak_value(s, max_len)
}

fn csv_stats_comment(col_names: Vec<&str>, data_rows: &[&str]) -> String {
    let rows: Vec<Vec<&str>> = data_rows
        .iter()
        .map(|r| r.split(',').collect())
        .filter(|r: &Vec<&str>| r.len() == col_names.len())
        .collect();
    if rows.is_empty() {
        return String::new();
    }
    let n = col_names.len();
    let mut numeric_cols = Vec::new();
    for ci in 0..n {
        if rows.iter().all(|r| r[ci].parse::<f64>().is_ok()) {
            numeric_cols.push(ci);
        }
    }
    if numeric_cols.is_empty() {
        return String::new();
    }
    let label_cols: Vec<usize> = (0..n).filter(|ci| !numeric_cols.contains(ci)).collect();
    let mut parts = Vec::new();
    for ci in numeric_cols {
        let vals: Vec<f64> = rows.iter().filter_map(|r| r[ci].parse().ok()).collect();
        let peak_val = vals.iter().copied().fold(f64::NAN, f64::max);
        let peak_row = rows
            .iter()
            .find(|r| r[ci].parse::<f64>().ok() == Some(peak_val))
            .unwrap();
        let labels: Vec<&str> = label_cols.iter().map(|&lc| peak_row[lc]).collect();
        parts.push(format!(
            "{}={}({})",
            col_names[ci],
            peak_val,
            labels.join(",")
        ));
    }
    format!("# peak: {}", parts.join(" "))
}

fn format_tool_args(args_str: &str) -> String {
    let s = args_str.trim();
    if s.is_empty() || s == "{}" {
        return String::new();
    }
    let Ok(obj) = serde_json::from_str::<Value>(s) else {
        return escape_aaak_value(s, 400);
    };
    let Some(map) = obj.as_object() else {
        return escape_aaak_value(s, 400);
    };
    let mut pairs = Vec::new();
    for (k, v) in map {
        let val = match v {
            Value::Array(a) => format!(
                "[{}]",
                a.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Value::Object(_) => serde_json::to_string(v).unwrap_or_default(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".into(),
            _ => v.to_string(),
        };
        pairs.push(format!("{k}={val}"));
    }
    pairs.join(";")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::FunctionCall;

    #[test]
    fn passthrough_aaak_context_preserves_existing_block() {
        let msg = ChatMessage::text("user", "[AAAK CTX]\nUSR: rate limits 3000 RPM\n[/AAAK CTX]");
        let out = passthrough_aaak_context(&[msg]);
        assert!(out.contains("rate limits 3000 RPM"));
    }

    #[test]
    fn encode_tool_round_flattens_json() {
        let tc = ToolCall {
            id: "1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "get_config".into(),
                arguments: "{}".into(),
            },
        };
        let tool_msg = ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(
                r#"{"pool_size":10,"timeout_ms":8500}"#.into(),
            )),
            tool_calls: None,
            tool_call_id: Some("1".into()),
            name: Some("get_config".into()),
            refusal: None,
            provider_blocks: None,
        };
        let mut id_to_name = std::collections::HashMap::new();
        id_to_name.insert("1".into(), "get_config".into());
        let out = AaakCompressor::encode_tool_round(&[tc], &[tool_msg], &id_to_name);
        assert!(out.starts_with("[AT]"));
        assert!(out.contains("pool_size=10"));
        assert!(out.contains("timeout_ms=8500"));
    }
}
