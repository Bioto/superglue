//! Serializable capture record shape (gzip NDJSON lines).

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CAPTURE_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureApi {
    Responses,
    ChatCompletions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRecord {
    pub schema: u32,
    pub request_id: String,
    pub ts_start: String,
    pub duration_ms: u64,
    pub user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    pub api: CaptureApi,
    pub model_requested: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_resolved: Option<String>,
    pub stream: bool,
    pub request: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
    pub sse: Vec<String>,
    pub sse_truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CaptureUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl CaptureRecord {
    pub fn to_ndjson_line(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut line = serde_json::to_vec(self)?;
        line.push(b'\n');
        Ok(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ndjson_line_ends_with_newline() {
        let record = CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: "req-1".into(),
            ts_start: "2026-01-01T00:00:00Z".into(),
            duration_ms: 10,
            user_id: "user-1".into(),
            key_id: Some("key-1".into()),
            api: CaptureApi::Responses,
            model_requested: "openai:gpt-4o".into(),
            model_resolved: Some("gpt-4o".into()),
            stream: true,
            request: json!({"model": "openai:gpt-4o"}),
            response: None,
            sse: vec!["event".into()],
            sse_truncated: false,
            usage: Some(CaptureUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
            }),
            cost_usd: Some(0.001),
            error: None,
        };
        let line = record.to_ndjson_line().unwrap();
        assert_eq!(line.last(), Some(&b'\n'));
        let parsed: Value = serde_json::from_slice(line.strip_suffix(b"\n").unwrap()).unwrap();
        assert_eq!(parsed["schema"], 1);
        assert_eq!(parsed["api"], "responses");
    }
}
