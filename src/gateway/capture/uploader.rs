//! Upload rotated spool files to S3 with retry/backoff.

use std::path::{Path, PathBuf};
use std::time::Duration;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_smithy_async::rt::sleep::TokioSleep;
use chrono::{DateTime, Utc};
use tracing::{info, warn};

use super::CaptureConfig;

const MAX_UPLOAD_ATTEMPTS: u32 = 5;

pub async fn build_s3_client(config: &CaptureConfig) -> Result<S3Client, String> {
    let mut loader =
        aws_config::defaults(aws_config::BehaviorVersion::latest()).sleep_impl(TokioSleep::new());
    if let Some(region) = &config.aws_region {
        loader = loader.region(aws_config::Region::new(region.clone()));
    }
    if let Some(endpoint) = &config.s3_endpoint {
        loader = loader.endpoint_url(endpoint);
    }
    let shared = loader.load().await;
    Ok(S3Client::new(&shared))
}

pub fn s3_object_key(prefix: &str, file_name: &str, now: DateTime<Utc>) -> String {
    let date = now.format("%Y-%m-%d");
    let hour = now.format("%H");
    format!("{prefix}/dt={date}/hour={hour}/{file_name}")
}

pub async fn upload_spool_file(
    client: &S3Client,
    config: &CaptureConfig,
    path: &Path,
) -> Result<(), String> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "spool path has no file name".to_string())?;
    let key = s3_object_key(&config.s3_prefix, file_name, Utc::now());
    let body = tokio::fs::read(path)
        .await
        .map_err(|e| format!("read spool file {}: {e}", path.display()))?;
    let mut delay = Duration::from_secs(1);
    for attempt in 1..=MAX_UPLOAD_ATTEMPTS {
        match client
            .put_object()
            .bucket(&config.s3_bucket)
            .key(&key)
            .content_type("application/gzip")
            .body(ByteStream::from(body.clone()))
            .send()
            .await
        {
            Ok(_) => {
                tokio::fs::remove_file(path).await.map_err(|e| {
                    format!("upload ok but failed to delete {}: {e}", path.display())
                })?;
                info!(
                    bucket = %config.s3_bucket,
                    key = %key,
                    "uploaded capture spool file"
                );
                return Ok(());
            }
            Err(err) if attempt < MAX_UPLOAD_ATTEMPTS => {
                warn!(
                    attempt,
                    max = MAX_UPLOAD_ATTEMPTS,
                    %err,
                    path = %path.display(),
                    "capture spool upload failed; retrying"
                );
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
            Err(err) => {
                return Err(format!(
                    "upload failed after {MAX_UPLOAD_ATTEMPTS} attempts for {}: {err}",
                    path.display()
                ));
            }
        }
    }
    Err("upload retries exhausted".into())
}

pub async fn upload_spool_file_background(client: S3Client, config: CaptureConfig, path: PathBuf) {
    if let Err(err) = upload_spool_file(&client, &config, &path).await {
        warn!(%err, path = %path.display(), "capture spool upload failed; leaving file on disk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_key_is_date_partitioned() {
        let ts = DateTime::parse_from_rfc3339("2026-08-06T15:04:05Z")
            .unwrap()
            .with_timezone(&Utc);
        let key = s3_object_key("gateway-capture", "gateway-abc-1.ndjson.gz", ts);
        assert_eq!(
            key,
            "gateway-capture/dt=2026-08-06/hour=15/gateway-abc-1.ndjson.gz"
        );
    }
}
