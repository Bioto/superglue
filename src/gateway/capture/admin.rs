//! Admin API for browsing captured LLM traffic (spool + S3 gzip NDJSON).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use aws_sdk_s3::Client as S3Client;
use axum::extract::{Path as AxumPath, Query, State};
use base64::Engine;
use chrono::{DateTime, Duration, Timelike, Utc};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};

use crate::gateway::GatewayState;
use crate::gateway::auth::Auth;
use crate::gateway::error::{GatewayError, GatewayResult};
use crate::gateway::routes::json_ok;

use super::CaptureRuntime;
use super::record::{CaptureApi, CaptureRecord, CaptureUsage};

const DEFAULT_LIST_LIMIT: u32 = 50;
const MAX_LIST_LIMIT: u32 = 200;
const MAX_HOUR_PARTITIONS: i64 = 72;
const CURSOR_VERSION: &str = "v1";

#[derive(Debug, Serialize)]
pub struct CaptureStatusResponse {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<CaptureConfigSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<CaptureStats>,
}

#[derive(Debug, Serialize)]
pub struct CaptureConfigSummary {
    pub s3_bucket: String,
    pub s3_prefix: String,
    pub spool_dir: String,
    pub rotate_bytes: u64,
    pub rotate_secs: u64,
    pub max_response_bytes: usize,
    pub exclude_users: Vec<String>,
    pub aws_region: Option<String>,
    pub s3_endpoint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CaptureStats {
    pub dropped_records: u64,
    pub pending_spool_files: u64,
    pub pending_spool_bytes: u64,
    pub channel_capacity: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureRecordSource {
    Spool,
    S3,
}

#[derive(Debug, Serialize)]
pub struct CaptureRecordSummary {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CaptureUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub sse_truncated: bool,
    pub source: CaptureRecordSource,
}

#[derive(Debug, Serialize)]
pub struct CaptureRecordsListResponse {
    pub records: Vec<CaptureRecordSummary>,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CaptureRecordDetailResponse {
    pub record: CaptureRecord,
    pub source: CaptureRecordSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CaptureRecordsQuery {
    pub user_id: Option<String>,
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: u32,
    pub cursor: Option<String>,
}

fn default_list_limit() -> u32 {
    DEFAULT_LIST_LIMIT
}

#[derive(Debug, Deserialize)]
pub struct CaptureRecordDetailQuery {
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, Clone)]
enum CaptureFile {
    Spool { path: PathBuf, modified: SystemTime },
    S3 { key: String, modified: SystemTime },
}

impl CaptureFile {
    fn modified(&self) -> SystemTime {
        match self {
            Self::Spool { modified, .. } | Self::S3 { modified, .. } => *modified,
        }
    }

    fn cursor_location(&self) -> String {
        match self {
            Self::Spool { path, .. } => encode_cursor_part(&path.display().to_string()),
            Self::S3 { key, .. } => encode_cursor_part(key),
        }
    }

    fn source(&self) -> CaptureRecordSource {
        match self {
            Self::Spool { .. } => CaptureRecordSource::Spool,
            Self::S3 { .. } => CaptureRecordSource::S3,
        }
    }

    fn source_path(&self) -> String {
        match self {
            Self::Spool { path, .. } => path.display().to_string(),
            Self::S3 { key, .. } => key.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScanCursor {
    file_index: usize,
    line_offset: u64,
}

struct RecordFilters<'a> {
    user_id: Option<&'a str>,
    request_id: Option<&'a str>,
    model: Option<&'a str>,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
}

impl CaptureRuntime {
    pub async fn status(&self) -> CaptureStatusResponse {
        let (pending_spool_files, pending_spool_bytes) =
            pending_spool_stats(&self.config.spool_dir).await;
        CaptureStatusResponse {
            enabled: true,
            config: Some(CaptureConfigSummary {
                s3_bucket: self.config.s3_bucket.clone(),
                s3_prefix: self.config.s3_prefix.clone(),
                spool_dir: self.config.spool_dir.display().to_string(),
                rotate_bytes: self.config.rotate_bytes,
                rotate_secs: self.config.rotate_secs,
                max_response_bytes: self.config.max_response_bytes,
                exclude_users: self.config.exclude_users.iter().cloned().collect(),
                aws_region: self.config.aws_region.clone(),
                s3_endpoint: self.config.s3_endpoint.clone(),
            }),
            stats: Some(CaptureStats {
                dropped_records: self.sink.dropped_count(),
                pending_spool_files,
                pending_spool_bytes,
                channel_capacity: super::DEFAULT_CHANNEL_CAPACITY as u64,
            }),
        }
    }

    pub async fn list_records(
        &self,
        query: CaptureRecordsQuery,
    ) -> Result<CaptureRecordsListResponse, GatewayError> {
        let limit = query.limit.clamp(1, MAX_LIST_LIMIT);
        let (from, to) = parse_time_range(query.from.as_deref(), query.to.as_deref())?;
        let filters = RecordFilters {
            user_id: query.user_id.as_deref(),
            request_id: query.request_id.as_deref(),
            model: query.model.as_deref(),
            from,
            to,
        };
        let files = collect_capture_files(&self.s3, &self.config, from, to).await;
        let start = parse_cursor(query.cursor.as_deref(), &files)?;
        scan_records(&self.s3, &self.config, &files, start, &filters, limit)
            .await
            .map_err(GatewayError::Internal)
    }
}

pub async fn capture_status(
    State(state): State<std::sync::Arc<GatewayState>>,
    Auth(_auth): Auth,
) -> GatewayResult<impl axum::response::IntoResponse> {
    match &state.capture {
        None => Ok(json_ok(CaptureStatusResponse {
            enabled: false,
            config: None,
            stats: None,
        })),
        Some(runtime) => Ok(json_ok(runtime.status().await)),
    }
}

pub async fn list_capture_records(
    State(state): State<std::sync::Arc<GatewayState>>,
    Auth(_auth): Auth,
    Query(query): Query<CaptureRecordsQuery>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let Some(runtime) = &state.capture else {
        return Ok(json_ok(CaptureRecordsListResponse {
            records: Vec::new(),
            truncated: false,
            next_cursor: None,
        }));
    };
    Ok(json_ok(runtime.list_records(query).await?))
}

pub async fn get_capture_record(
    State(state): State<std::sync::Arc<GatewayState>>,
    Auth(_auth): Auth,
    AxumPath(request_id): AxumPath<String>,
    Query(query): Query<CaptureRecordDetailQuery>,
) -> GatewayResult<impl axum::response::IntoResponse> {
    let Some(runtime) = &state.capture else {
        return Err(GatewayError::not_found("capture not enabled"));
    };
    let (from_dt, to_dt) = parse_time_range(query.from.as_deref(), query.to.as_deref())?;
    let detail = find_record_detail(runtime, &request_id, from_dt, to_dt)
        .await
        .map_err(GatewayError::Internal)?;
    detail
        .ok_or_else(|| GatewayError::not_found(format!("capture record {request_id} not found")))
        .map(|record| json_ok(record))
}

async fn find_record_detail(
    runtime: &CaptureRuntime,
    request_id: &str,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Option<CaptureRecordDetailResponse>, String> {
    let files = collect_capture_files(&runtime.s3, &runtime.config, from, to).await;
    for file in &files {
        let lines = match read_capture_file_lines(&runtime.s3, &runtime.config, file).await {
            Ok(lines) => lines,
            Err(_) => continue,
        };
        for line in lines {
            let record: CaptureRecord =
                serde_json::from_str(&line).map_err(|e| format!("parse capture record: {e}"))?;
            if record.request_id != request_id {
                continue;
            }
            if !record_in_time_range(&record, from, to) {
                continue;
            }
            return Ok(Some(CaptureRecordDetailResponse {
                record,
                source: file.source(),
                source_path: Some(file.source_path()),
            }));
        }
    }
    Ok(None)
}

async fn pending_spool_stats(spool_dir: &Path) -> (u64, u64) {
    let mut files = 0u64;
    let mut bytes = 0u64;
    let Ok(mut rd) = tokio::fs::read_dir(spool_dir).await else {
        return (files, bytes);
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if !is_spool_file(&path) {
            continue;
        }
        if let Ok(meta) = entry.metadata().await {
            files += 1;
            bytes += meta.len();
        }
    }
    (files, bytes)
}

fn parse_time_range(
    from: Option<&str>,
    to: Option<&str>,
) -> Result<(DateTime<Utc>, DateTime<Utc>), GatewayError> {
    let to_dt = match to {
        Some(raw) => parse_rfc3339(raw).map_err(GatewayError::bad_request)?,
        None => Utc::now(),
    };
    let from_dt = match from {
        Some(raw) => parse_rfc3339(raw).map_err(GatewayError::bad_request)?,
        None => to_dt - Duration::hours(24),
    };
    if from_dt > to_dt {
        return Err(GatewayError::bad_request("from must be before to"));
    }
    Ok((from_dt, to_dt))
}

fn parse_rfc3339(raw: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| format!("invalid RFC3339 timestamp: {e}"))
}

fn hour_prefixes(s3_prefix: &str, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<String> {
    let mut prefixes = Vec::new();
    let mut current = to
        .with_minute(0)
        .and_then(|t| t.with_second(0))
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(to);
    let from_floor = from
        .with_minute(0)
        .and_then(|t| t.with_second(0))
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(from);
    let mut scanned = 0i64;
    while current >= from_floor && scanned < MAX_HOUR_PARTITIONS {
        prefixes.push(format!(
            "{}/dt={}/hour={:02}/",
            s3_prefix,
            current.format("%Y-%m-%d"),
            current.hour()
        ));
        current -= Duration::hours(1);
        scanned += 1;
    }
    prefixes
}

async fn collect_capture_files(
    s3: &S3Client,
    config: &super::CaptureConfig,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Vec<CaptureFile> {
    let mut files = list_spool_capture_files(&config.spool_dir).await;
    files.extend(list_s3_capture_files(s3, config, from, to).await);
    files.sort_by_key(|f| std::cmp::Reverse(f.modified()));
    files
}

async fn list_spool_capture_files(spool_dir: &Path) -> Vec<CaptureFile> {
    let mut files = Vec::new();
    let Ok(mut rd) = tokio::fs::read_dir(spool_dir).await else {
        return files;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if !is_spool_file(&path) {
            continue;
        }
        let modified = entry
            .metadata()
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(UNIX_EPOCH);
        files.push(CaptureFile::Spool { path, modified });
    }
    files
}

async fn list_s3_capture_files(
    s3: &S3Client,
    config: &super::CaptureConfig,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Vec<CaptureFile> {
    let mut files = Vec::new();
    for prefix in hour_prefixes(&config.s3_prefix, from, to) {
        let mut continuation = None;
        loop {
            let mut req = s3
                .list_objects_v2()
                .bucket(&config.s3_bucket)
                .prefix(&prefix);
            if let Some(token) = continuation.as_ref() {
                req = req.continuation_token(token);
            }
            let resp = match req.send().await {
                Ok(resp) => resp,
                Err(_) => break,
            };
            for obj in resp.contents() {
                let Some(key) = obj.key() else { continue };
                if !key.ends_with(".ndjson.gz") {
                    continue;
                }
                let modified = obj
                    .last_modified()
                    .and_then(|t| {
                        t.secs()
                            .try_into()
                            .ok()
                            .map(|secs| UNIX_EPOCH + std::time::Duration::from_secs(secs))
                    })
                    .unwrap_or(UNIX_EPOCH);
                files.push(CaptureFile::S3 {
                    key: key.to_string(),
                    modified,
                });
            }
            continuation = resp.next_continuation_token().map(str::to_string);
            if continuation.is_none() {
                break;
            }
        }
    }
    files
}

fn is_spool_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".ndjson.gz"))
}

fn encode_cursor_part(raw: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw.as_bytes())
}

fn decode_cursor_part(encoded: &str) -> Option<String> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .ok()?;
    String::from_utf8(bytes).ok()
}

fn format_cursor(file: &CaptureFile, line_offset: u64) -> String {
    let kind = match file {
        CaptureFile::Spool { .. } => "spool",
        CaptureFile::S3 { .. } => "s3",
    };
    format!(
        "{CURSOR_VERSION}:{kind}:{}:{line_offset}",
        file.cursor_location()
    )
}

fn parse_cursor(cursor: Option<&str>, files: &[CaptureFile]) -> Result<ScanCursor, GatewayError> {
    let Some(raw) = cursor else {
        return Ok(ScanCursor {
            file_index: 0,
            line_offset: 0,
        });
    };
    let parts: Vec<&str> = raw.split(':').collect();
    if parts.len() != 4 || parts[0] != CURSOR_VERSION {
        return Err(GatewayError::bad_request("invalid cursor"));
    }
    let kind = parts[1];
    let location = decode_cursor_part(parts[2])
        .ok_or_else(|| GatewayError::bad_request("invalid cursor location"))?;
    let line_offset = parts[3]
        .parse::<u64>()
        .map_err(|_| GatewayError::bad_request("invalid cursor line offset"))?;
    let file_index = files
        .iter()
        .position(|file| match (kind, file) {
            ("spool", CaptureFile::Spool { path, .. }) => path.display().to_string() == location,
            ("s3", CaptureFile::S3 { key, .. }) => key == &location,
            _ => false,
        })
        .ok_or_else(|| GatewayError::bad_request("cursor file no longer available"))?;
    Ok(ScanCursor {
        file_index,
        line_offset,
    })
}

async fn scan_records(
    s3: &S3Client,
    config: &super::CaptureConfig,
    files: &[CaptureFile],
    mut start: ScanCursor,
    filters: &RecordFilters<'_>,
    limit: u32,
) -> Result<CaptureRecordsListResponse, String> {
    let mut records = Vec::new();
    let mut truncated = false;
    let mut next_cursor: Option<String> = None;

    'files: for file_index in start.file_index..files.len() {
        let file = &files[file_index];
        let lines = match read_capture_file_lines(s3, config, file).await {
            Ok(lines) => lines,
            Err(_) => continue,
        };
        let line_count = lines.len();
        let skip = if file_index == start.file_index {
            start.line_offset
        } else {
            0
        };
        for (line_idx, line) in lines.into_iter().enumerate().skip(skip as usize) {
            let record: CaptureRecord = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(_) => continue,
            };
            if !record_matches(&record, filters) {
                continue;
            }
            records.push(record_to_summary(&record, file.source()));
            if records.len() >= limit as usize {
                let next_line = line_idx as u64 + 1;
                if (next_line as usize) < line_count {
                    next_cursor = Some(format_cursor(file, next_line));
                } else if file_index + 1 < files.len() {
                    next_cursor = Some(format_cursor(&files[file_index + 1], 0));
                }
                truncated = next_cursor.is_some();
                break 'files;
            }
        }
        start.line_offset = 0;
    }

    Ok(CaptureRecordsListResponse {
        records,
        truncated,
        next_cursor,
    })
}

fn record_to_summary(record: &CaptureRecord, source: CaptureRecordSource) -> CaptureRecordSummary {
    CaptureRecordSummary {
        request_id: record.request_id.clone(),
        ts_start: record.ts_start.clone(),
        duration_ms: record.duration_ms,
        user_id: record.user_id.clone(),
        key_id: record.key_id.clone(),
        api: record.api,
        model_requested: record.model_requested.clone(),
        model_resolved: record.model_resolved.clone(),
        stream: record.stream,
        usage: record.usage.clone(),
        cost_usd: record.cost_usd,
        error: record.error.clone(),
        sse_truncated: record.sse_truncated,
        source,
    }
}

fn record_matches(record: &CaptureRecord, filters: &RecordFilters<'_>) -> bool {
    if !record_in_time_range(record, filters.from, filters.to) {
        return false;
    }
    if let Some(user_id) = filters.user_id {
        if record.user_id != user_id {
            return false;
        }
    }
    if let Some(request_id) = filters.request_id {
        if record.request_id != request_id {
            return false;
        }
    }
    if let Some(model) = filters.model {
        let haystack = format!(
            "{} {}",
            record.model_requested,
            record.model_resolved.as_deref().unwrap_or("")
        );
        if !haystack
            .to_ascii_lowercase()
            .contains(&model.to_ascii_lowercase())
        {
            return false;
        }
    }
    true
}

fn record_in_time_range(record: &CaptureRecord, from: DateTime<Utc>, to: DateTime<Utc>) -> bool {
    match parse_rfc3339(&record.ts_start) {
        Ok(ts) => ts >= from && ts <= to,
        Err(_) => true,
    }
}

async fn read_capture_file_lines(
    s3: &S3Client,
    config: &super::CaptureConfig,
    file: &CaptureFile,
) -> Result<Vec<String>, String> {
    let bytes = match file {
        CaptureFile::Spool { path, .. } => tokio::fs::read(path)
            .await
            .map_err(|e| format!("read spool file {}: {e}", path.display()))?,
        CaptureFile::S3 { key, .. } => {
            let resp = s3
                .get_object()
                .bucket(&config.s3_bucket)
                .key(key)
                .send()
                .await
                .map_err(|e| format!("get s3 object {key}: {e}"))?;
            let data = resp
                .body
                .collect()
                .await
                .map_err(|e| format!("read s3 body {key}: {e}"))?
                .into_bytes();
            data.to_vec()
        }
    };
    decode_gzip_lines(&bytes)
}

fn decode_gzip_lines(bytes: &[u8]) -> Result<Vec<String>, String> {
    let decoder = GzDecoder::new(bytes);
    let reader = BufReader::new(decoder);
    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read gzip line: {e}"))?;
        if !line.trim().is_empty() {
            lines.push(line);
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::capture::record::CAPTURE_SCHEMA;
    use serde_json::json;
    use std::collections::HashSet;
    use std::io::Write;

    fn sample_record(request_id: &str, user_id: &str, ts: &str) -> CaptureRecord {
        CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: request_id.into(),
            ts_start: ts.into(),
            duration_ms: 10,
            user_id: user_id.into(),
            key_id: None,
            api: CaptureApi::ChatCompletions,
            model_requested: "openai:gpt-4o".into(),
            model_resolved: Some("gpt-4o".into()),
            stream: false,
            request: json!({"model": "openai:gpt-4o"}),
            response: Some(json!({"ok": true})),
            sse: Vec::new(),
            sse_truncated: false,
            usage: Some(CaptureUsage {
                prompt_tokens: 1,
                completion_tokens: 2,
            }),
            cost_usd: Some(0.001),
            error: None,
        }
    }

    fn write_spool_file(path: &Path, records: &[CaptureRecord]) {
        let file = std::fs::File::create(path).unwrap();
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
        for record in records {
            let mut line = serde_json::to_vec(record).unwrap();
            line.push(b'\n');
            encoder.write_all(&line).unwrap();
        }
        encoder.finish().unwrap();
    }

    #[tokio::test]
    async fn list_records_from_spool_with_filters() {
        let dir = tempfile::tempdir().unwrap();
        let spool_dir = dir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();
        let path = spool_dir.join("gateway-test-1.ndjson.gz");
        write_spool_file(
            &path,
            &[
                sample_record("req-1", "alice", "2026-08-06T12:00:00Z"),
                sample_record("req-2", "bob", "2026-08-06T12:01:00Z"),
            ],
        );

        let config = super::super::CaptureConfig {
            s3_bucket: "unused".into(),
            s3_prefix: "gateway-capture".into(),
            spool_dir: spool_dir.clone(),
            rotate_bytes: 1024,
            rotate_secs: 60,
            max_response_bytes: 1024,
            exclude_users: HashSet::new(),
            aws_region: None,
            s3_endpoint: Some("http://127.0.0.1:9".into()),
        };
        let s3 = super::super::uploader::build_s3_client(&config)
            .await
            .expect("s3 client");
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let runtime = CaptureRuntime {
            sink: std::sync::Arc::new(super::super::CaptureSink {
                tx,
                dropped: std::sync::atomic::AtomicU64::new(0),
                exclude_users: HashSet::new(),
                max_response_bytes: 1024,
            }),
            config,
            s3,
        };

        let all = runtime
            .list_records(CaptureRecordsQuery {
                user_id: None,
                request_id: None,
                model: None,
                from: Some("2026-08-06T00:00:00Z".into()),
                to: Some("2026-08-06T23:59:59Z".into()),
                limit: 10,
                cursor: None,
            })
            .await
            .unwrap();
        assert_eq!(all.records.len(), 2);

        let alice = runtime
            .list_records(CaptureRecordsQuery {
                user_id: Some("alice".into()),
                request_id: None,
                model: None,
                from: None,
                to: None,
                limit: 10,
                cursor: None,
            })
            .await
            .unwrap();
        assert_eq!(alice.records.len(), 1);
        assert_eq!(alice.records[0].request_id, "req-1");
    }
}
