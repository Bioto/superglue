//! Optional LLM traffic capture: gzip NDJSON spool + background S3 upload.

mod admin;
mod record;
mod spool;
mod tap;
mod uploader;

pub use admin::{
    capture_status, get_capture_record, list_capture_records, CaptureStatusResponse,
};
pub use record::{CaptureApi, CaptureRecord};
pub use tap::{CaptureTap, CaptureTapContext};

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use aws_sdk_s3::Client as S3Client;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::gateway::error::GatewayError;

use self::spool::{recover_orphan_spool_files, SpoolWriter};
use self::uploader::{build_s3_client, upload_spool_file_background};

pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 256;

const DEFAULT_S3_PREFIX: &str = "gateway-capture";
const DEFAULT_ROTATE_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_ROTATE_SECS: u64 = 300;
const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Runtime configuration for capture (enabled when `s3_bucket` is set).
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub s3_bucket: String,
    pub s3_prefix: String,
    pub spool_dir: PathBuf,
    pub rotate_bytes: u64,
    pub rotate_secs: u64,
    pub max_response_bytes: usize,
    pub exclude_users: HashSet<String>,
    pub aws_region: Option<String>,
    pub s3_endpoint: Option<String>,
}

impl CaptureConfig {
    /// Build capture config from environment variables; returns `None` when disabled.
    #[must_use]
    pub fn from_env(db_path: &Path) -> Option<Self> {
        let bucket = std::env::var("SUPERGLUE_CAPTURE_S3_BUCKET")
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        let spool_dir = std::env::var("SUPERGLUE_CAPTURE_SPOOL_DIR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                db_path
                    .parent()
                    .map(|p| p.join("capture-spool"))
                    .unwrap_or_else(|| PathBuf::from("capture-spool"))
            });
        Some(Self {
            s3_bucket: bucket.trim().to_string(),
            s3_prefix: std::env::var("SUPERGLUE_CAPTURE_S3_PREFIX")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_S3_PREFIX.to_string()),
            spool_dir,
            rotate_bytes: parse_u64_env("SUPERGLUE_CAPTURE_ROTATE_BYTES", DEFAULT_ROTATE_BYTES),
            rotate_secs: parse_u64_env("SUPERGLUE_CAPTURE_ROTATE_SECS", DEFAULT_ROTATE_SECS),
            max_response_bytes: parse_usize_env(
                "SUPERGLUE_CAPTURE_MAX_RESPONSE_BYTES",
                DEFAULT_MAX_RESPONSE_BYTES,
            ),
            exclude_users: parse_exclude_users(
                std::env::var("SUPERGLUE_CAPTURE_EXCLUDE_USERS").ok().as_deref(),
            ),
            aws_region: std::env::var("AWS_REGION")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            s3_endpoint: std::env::var("SUPERGLUE_CAPTURE_S3_ENDPOINT")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        })
    }

    /// Build from explicit CLI overrides layered on top of env defaults.
    #[must_use]
    pub fn from_serve_args(
        db_path: &Path,
        s3_bucket: Option<String>,
        s3_prefix: Option<String>,
    ) -> Option<Self> {
        let bucket = s3_bucket
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::env::var("SUPERGLUE_CAPTURE_S3_BUCKET")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })?;
        let mut config = Self::from_env(db_path).unwrap_or_else(|| {
            let spool_dir = db_path
                .parent()
                .map(|p| p.join("capture-spool"))
                .unwrap_or_else(|| PathBuf::from("capture-spool"));
            Self {
                s3_bucket: bucket.clone(),
                s3_prefix: DEFAULT_S3_PREFIX.to_string(),
                spool_dir,
                rotate_bytes: DEFAULT_ROTATE_BYTES,
                rotate_secs: DEFAULT_ROTATE_SECS,
                max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
                exclude_users: HashSet::new(),
                aws_region: None,
                s3_endpoint: None,
            }
        });
        config.s3_bucket = bucket;
        if let Some(prefix) = s3_prefix.filter(|s| !s.trim().is_empty()) {
            config.s3_prefix = prefix;
        }
        Some(config)
    }
}

fn parse_u64_env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn parse_exclude_users(raw: Option<&str>) -> HashSet<String> {
    raw.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// Non-blocking capture sink; drops records when the writer falls behind.
pub struct CaptureSink {
    tx: mpsc::Sender<CaptureRecord>,
    dropped: AtomicU64,
    exclude_users: HashSet<String>,
    max_response_bytes: usize,
}

impl CaptureSink {
    #[must_use]
    pub fn max_response_bytes(&self) -> usize {
        self.max_response_bytes
    }

    #[must_use]
    pub fn should_exclude(&self, user_id: &str) -> bool {
        self.exclude_users.contains(user_id)
    }

    pub fn try_send(&self, record: CaptureRecord) {
        if self.should_exclude(&record.user_id) {
            return;
        }
        if self.tx.try_send(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Live capture pipeline: sink for writes plus config/S3 client for admin reads.
pub struct CaptureRuntime {
    sink: Arc<CaptureSink>,
    pub(crate) config: CaptureConfig,
    pub(crate) s3: S3Client,
}

impl CaptureRuntime {
    #[must_use]
    pub fn sink(&self) -> Arc<CaptureSink> {
        self.sink.clone()
    }
}

/// Start the background writer + uploader tasks.
pub async fn spawn_capture_pipeline(
    config: CaptureConfig,
) -> Result<Arc<CaptureRuntime>, GatewayError> {
    let (tx, rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let writer_id = Uuid::new_v4().to_string();
    let max_response_bytes = config.max_response_bytes;
    let exclude_users = config.exclude_users.clone();
    let sink = Arc::new(CaptureSink {
        tx,
        dropped: AtomicU64::new(0),
        exclude_users,
        max_response_bytes,
    });

    let s3_client = build_s3_client(&config)
        .await
        .map_err(|e| GatewayError::Internal(format!("capture S3 client: {e}")))?;

    let runtime = Arc::new(CaptureRuntime {
        sink: sink.clone(),
        config: config.clone(),
        s3: s3_client.clone(),
    });

    info!(
        bucket = %config.s3_bucket,
        prefix = %config.s3_prefix,
        spool_dir = %config.spool_dir.display(),
        "gateway LLM capture enabled"
    );

    tokio::spawn(capture_writer_loop(
        rx,
        config,
        writer_id,
        s3_client,
    ));

    Ok(runtime)
}

async fn capture_writer_loop(
    mut rx: mpsc::Receiver<CaptureRecord>,
    config: CaptureConfig,
    writer_id: String,
    s3_client: aws_sdk_s3::Client,
) {
    let mut spool = match SpoolWriter::open(
        &config.spool_dir,
        &writer_id,
        config.rotate_bytes,
        config.rotate_secs,
    ) {
        Ok(writer) => writer,
        Err(err) => {
            warn!(%err, "failed to open capture spool; capture disabled");
            return;
        }
    };

    for orphan in recover_orphan_spool_files(&config.spool_dir) {
        if orphan == spool.current_path() {
            continue;
        }
        let cfg = config.clone();
        let client = s3_client.clone();
        tokio::spawn(upload_spool_file_background(client, cfg, orphan));
    }

    while let Some(record) = rx.recv().await {
        if let Err(err) = spool.append(&record) {
            warn!(%err, "failed to append capture record");
            continue;
        }
        if spool.should_rotate() {
            match spool.rotate() {
                Ok(rotated) => {
                    let cfg = config.clone();
                    let client = s3_client.clone();
                    tokio::spawn(upload_spool_file_background(client, cfg, rotated));
                }
                Err(err) => warn!(%err, "failed to rotate capture spool"),
            }
        }
    }

    if spool.has_data()
        && let Ok(rotated) = spool.rotate()
    {
        upload_spool_file_background(s3_client, config, rotated).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::capture::record::{CaptureApi, CaptureRecord, CAPTURE_SCHEMA};

    #[test]
    fn exclude_users_parsed() {
        let users = parse_exclude_users(Some("alice, bob ,,carol"));
        assert_eq!(users.len(), 3);
        assert!(users.contains("alice"));
        assert!(users.contains("bob"));
        assert!(users.contains("carol"));
    }

    #[test]
    fn sink_drops_for_excluded_user() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut excluded = HashSet::new();
        excluded.insert("skip-me".into());
        let sink = CaptureSink {
            tx,
            dropped: AtomicU64::new(0),
            exclude_users: excluded,
            max_response_bytes: 1024,
        };
        assert!(sink.should_exclude("skip-me"));
        sink.try_send(CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: "r".into(),
            ts_start: "t".into(),
            duration_ms: 0,
            user_id: "skip-me".into(),
            key_id: None,
            api: CaptureApi::Responses,
            model_requested: "m".into(),
            model_resolved: None,
            stream: false,
            request: serde_json::json!({}),
            response: None,
            sse: Vec::new(),
            sse_truncated: false,
            usage: None,
            cost_usd: None,
            error: None,
        });
        assert!(rx.try_recv().is_err());
    }
}
