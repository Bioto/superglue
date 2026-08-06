//! Local gzip NDJSON spool with size/age rotation.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use flate2::Compression;
use flate2::write::GzEncoder;
use tracing::{info, warn};

use super::record::CaptureRecord;

const SPOOL_SUFFIX: &str = ".ndjson.gz";
const SPOOL_PREFIX: &str = "gateway-";

#[derive(Debug)]
pub struct SpoolWriter {
    spool_dir: PathBuf,
    writer_id: String,
    rotate_bytes: u64,
    rotate_secs: u64,
    current_path: PathBuf,
    encoder: Option<GzEncoder<File>>,
    bytes_written: u64,
    opened_at: Instant,
}

impl SpoolWriter {
    pub fn open(
        spool_dir: &Path,
        writer_id: &str,
        rotate_bytes: u64,
        rotate_secs: u64,
    ) -> io::Result<Self> {
        fs::create_dir_all(spool_dir)?;
        let (path, file) = Self::create_spool_file(spool_dir, writer_id)?;
        let encoder = GzEncoder::new(file, Compression::fast());
        Ok(Self {
            spool_dir: spool_dir.to_path_buf(),
            writer_id: writer_id.to_string(),
            rotate_bytes,
            rotate_secs,
            current_path: path,
            encoder: Some(encoder),
            bytes_written: 0,
            opened_at: Instant::now(),
        })
    }

    fn create_spool_file(spool_dir: &Path, writer_id: &str) -> io::Result<(PathBuf, File)> {
        let unique = uuid::Uuid::new_v4();
        let name = format!("{SPOOL_PREFIX}{writer_id}-{unique}{SPOOL_SUFFIX}");
        let path = spool_dir.join(name);
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        Ok((path, file))
    }

    pub fn current_path(&self) -> &Path {
        &self.current_path
    }

    pub fn append(&mut self, record: &CaptureRecord) -> io::Result<()> {
        let line = record
            .to_ndjson_line()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| io::Error::other("spool encoder is closed"))?;
        encoder.write_all(&line)?;
        self.bytes_written += line.len() as u64;
        Ok(())
    }

    pub fn should_rotate(&self) -> bool {
        self.bytes_written >= self.rotate_bytes
            || self.opened_at.elapsed() >= Duration::from_secs(self.rotate_secs)
    }

    #[must_use]
    pub fn has_data(&self) -> bool {
        self.bytes_written > 0
    }

    /// Finish the current file and return its path for upload.
    pub fn rotate(&mut self) -> io::Result<PathBuf> {
        let rotated = self.current_path.clone();
        let encoder = self
            .encoder
            .take()
            .ok_or_else(|| io::Error::other("spool encoder is closed"))?;
        encoder
            .finish()
            .map_err(|e| io::Error::other(format!("gzip finish: {e}")))?;
        let (path, file) = Self::create_spool_file(&self.spool_dir, &self.writer_id)?;
        self.current_path = path;
        self.encoder = Some(GzEncoder::new(file, Compression::fast()));
        self.bytes_written = 0;
        self.opened_at = Instant::now();
        Ok(rotated)
    }

    /// List orphaned spool files from prior runs (excluding the active writer prefix match is not needed — all closed files).
    pub fn list_orphan_files(spool_dir: &Path) -> io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        if !spool_dir.is_dir() {
            return Ok(out);
        }
        for entry in fs::read_dir(spool_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(SPOOL_PREFIX) && n.ends_with(SPOOL_SUFFIX))
            {
                out.push(path);
            }
        }
        out.sort();
        Ok(out)
    }
}

pub fn recover_orphan_spool_files(spool_dir: &Path) -> Vec<PathBuf> {
    match SpoolWriter::list_orphan_files(spool_dir) {
        Ok(files) => {
            if !files.is_empty() {
                info!(count = files.len(), "found orphaned capture spool files");
            }
            files
        }
        Err(err) => {
            warn!(%err, "failed to scan capture spool dir for orphans");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::capture::record::{CaptureApi, CaptureRecord, CAPTURE_SCHEMA};
    use serde_json::json;

    fn sample_record(id: &str) -> CaptureRecord {
        CaptureRecord {
            schema: CAPTURE_SCHEMA,
            request_id: id.into(),
            ts_start: "2026-01-01T00:00:00Z".into(),
            duration_ms: 1,
            user_id: "user".into(),
            key_id: None,
            api: CaptureApi::ChatCompletions,
            model_requested: "openai:gpt-4o".into(),
            model_resolved: None,
            stream: false,
            request: json!({}),
            response: None,
            sse: Vec::new(),
            sse_truncated: false,
            usage: None,
            cost_usd: None,
            error: None,
        }
    }

    #[test]
    fn rotates_when_byte_limit_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            SpoolWriter::open(dir.path(), "test-writer", 64, 3600).expect("open spool");
        let big = "x".repeat(128);
        let mut record = sample_record("big");
        record.request = json!({ "payload": big });
        writer.append(&record).unwrap();
        assert!(writer.should_rotate());
        let rotated = writer.rotate().unwrap();
        assert!(rotated.exists());
        assert_ne!(rotated, writer.current_path());
    }

    #[test]
    fn lists_orphan_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gateway-dead-beef-1.ndjson.gz");
        File::create(&path).unwrap();
        let orphans = SpoolWriter::list_orphan_files(dir.path()).unwrap();
        assert_eq!(orphans, vec![path]);
    }
}
