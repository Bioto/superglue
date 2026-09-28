//! Observability initialisation for the superglue core.
//!
//! Call [`init_tracing`] once at startup (typically from `main`) to wire up:
//!
//! - A `tracing_subscriber::fmt` layer with [`scrub::ScrubFields`] that redacts
//!   sensitive field names ([`scrub::SENSITIVE_FIELDS`]) from console/log output.
//! - Optionally (when compiled with the `otlp` feature) an OpenTelemetry OTLP
//!   layer that exports spans to a collector
//!   (configure via [`TelemetryConfig::otlp_endpoint`]).
//!
//! [`metrics`] initialisation is handled separately via [`metrics::init_metrics`].
//!
//! # Levels and structure
//!
//! - **`ERROR` / `WARN`**: failures, retries, skipped tools.
//! - **`INFO`**: request lifecycle (start/finish, model, `request_id`, round counts).
//! - **`DEBUG`**: connection and stream milestones.
//! - **`TRACE`**: high-volume detail (per-SSE-chunk, per-token).
//!
//! Prefer structured fields (`model`, `request_id`, `error = %e`). Do not log secrets;
//! [`ChatOptions`] omits `api_key` from `Debug`. The fmt layer scrubs known sensitive
//! **field names** in output ([`scrub::SENSITIVE_FIELDS`]).
//!
//! Embedders should install their own subscriber or call [`init_tracing`] once;
//! [`init_tracing`] uses `try_init` and ignores duplicate registration.
//!
//! # Example
//!
//! ```rust,no_run
//! use superglue::telemetry::{LogFormat, ScrubMode, TelemetryConfig, init_tracing};
//!
//! init_tracing(TelemetryConfig {
//!     otlp_endpoint: Some("http://localhost:4318".to_string()),
//!     log_level: "info".to_string(),
//!     scrub_mode: ScrubMode::Redact,
//!     log_format: LogFormat::Pretty,
//! });
//! ```

pub mod metrics;
pub mod scrub;

use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
};

use tracing_subscriber::{
    EnvFilter, Layer, Registry, fmt::MakeWriter, layer::SubscriberExt, util::SubscriberInitExt,
};

use scrub::{ScrubFields, ScrubJsonFields};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Console / stderr log line format for the fmt layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// Human-readable lines (default).
    #[default]
    Pretty,
    /// One JSON object per line (useful for log aggregators).
    Json,
}

/// Configuration for [`init_tracing`].
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// When set, spans are exported via OTLP to this endpoint.
    ///
    /// Example: `"http://localhost:4318"` (OpenTelemetry Collector default).
    /// Set to `None` to disable OTLP export (local console only).
    ///
    /// Requires the `otlp` Cargo feature. When the feature is absent this
    /// field is silently ignored.
    pub otlp_endpoint: Option<String>,

    /// Log level filter string, forwarded to [`EnvFilter`].
    ///
    /// Overridden by the `RUST_LOG` environment variable when that is set.
    /// Example: `"info"`, `"superglue=debug,warn"`.
    pub log_level: String,

    /// How sensitive field values are handled in formatted console output.
    ///
    /// - [`ScrubMode::Redact`] (default) — replaces values with `[REDACTED]`.
    ///   Use in production.
    /// - [`ScrubMode::Hash`] — replaces values with `[HASH:xxxxxxxx]`.
    ///   Useful for correlating events in debug sessions without revealing content.
    /// - [`ScrubMode::Allow`] — passes values through unchanged.
    ///   Only for local development.
    pub scrub_mode: ScrubMode,

    /// Pretty (multi-line) vs JSON lines for the fmt subscriber layer.
    pub log_format: LogFormat,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        TelemetryConfig {
            otlp_endpoint: None,
            log_level: "info".to_string(),
            scrub_mode: ScrubMode::Redact,
            log_format: LogFormat::Pretty,
        }
    }
}

pub use scrub::ScrubMode;

// ---------------------------------------------------------------------------
// Public initialisation
// ---------------------------------------------------------------------------

/// Initialise the global `tracing` subscriber.
///
/// Must be called exactly once before any `tracing::info!` etc. macros fire.
/// The function is a no-op if a global subscriber is already installed
/// (e.g. in tests).
///
/// When compiled with the `otlp` Cargo feature **and** `config.otlp_endpoint` is
/// `Some`, a [`tracing_opentelemetry`] layer is added that exports spans to the
/// specified OTLP/HTTP endpoint (e.g. an OpenTelemetry Collector). The `fmt`
/// layer (with optional scrubbing) is always installed regardless.
pub fn init_tracing(config: TelemetryConfig) {
    init_tracing_with_file(config, None);
}

/// Initialise tracing with an optional second fmt output file.
///
/// The file writer opens the path for each event. This keeps log rotation owned by
/// the host process and avoids retaining a handle across a rotation.
pub fn init_tracing_with_file(config: TelemetryConfig, log_path: Option<PathBuf>) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    let writer = DualMakeWriter { path: log_path };
    let fmt_layer: Box<dyn Layer<Registry> + Send + Sync> =
        boxed_fmt_layer_with_writer(&config, writer.clone());

    #[cfg(feature = "otlp")]
    if let Some(ref endpoint) = config.otlp_endpoint {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_otlp::{SpanExporter, WithExportConfig};
        use opentelemetry_sdk::trace::SdkTracerProvider;
        use tracing_opentelemetry::OpenTelemetryLayer;

        match SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint.as_str())
            .build()
        {
            Ok(exporter) => {
                let provider = SdkTracerProvider::builder()
                    .with_batch_exporter(exporter)
                    .build();
                let tracer = provider.tracer("superglue");
                let otel_layer = OpenTelemetryLayer::new(tracer).boxed();
                let _ = tracing_subscriber::registry()
                    .with(fmt_layer)
                    .with(otel_layer)
                    .with(filter)
                    .try_init();
                return;
            }
            Err(e) => {
                // Fall through to fmt-only; warn after the subscriber is installed.
                let fmt_layer: Box<dyn Layer<Registry> + Send + Sync> =
                    boxed_fmt_layer_with_writer(&config, writer);
                let _ = tracing_subscriber::registry()
                    .with(fmt_layer)
                    .with(filter)
                    .try_init();
                tracing::warn!(error = %e, "failed to build OTLP exporter; using fmt-only tracing");
                return;
            }
        }
    }

    let _ = tracing_subscriber::registry()
        .with(fmt_layer)
        .with(filter)
        .try_init();
}

fn boxed_fmt_layer_with_writer<W>(
    config: &TelemetryConfig,
    writer: W,
) -> Box<dyn Layer<Registry> + Send + Sync>
where
    W: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    match (config.log_format, config.scrub_mode) {
        (LogFormat::Json, scrub::ScrubMode::Allow) => tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_writer(writer)
            .boxed(),
        (LogFormat::Pretty, scrub::ScrubMode::Allow) => tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_writer(writer)
            .boxed(),
        (LogFormat::Json, mode) => tracing_subscriber::fmt::layer()
            .json()
            .fmt_fields(ScrubJsonFields(mode))
            .with_target(true)
            .with_writer(writer)
            .boxed(),
        (LogFormat::Pretty, mode) => tracing_subscriber::fmt::layer()
            .fmt_fields(ScrubFields(mode))
            .with_target(true)
            .with_writer(writer)
            .boxed(),
    }
}

#[derive(Clone)]
struct DualMakeWriter {
    path: Option<PathBuf>,
}

struct DualWriter {
    stderr: io::Stderr,
    file: Option<std::fs::File>,
}

impl<'a> MakeWriter<'a> for DualMakeWriter {
    type Writer = DualWriter;

    fn make_writer(&'a self) -> Self::Writer {
        let file = self
            .path
            .as_ref()
            .and_then(|path| OpenOptions::new().create(true).append(true).open(path).ok());
        DualWriter {
            stderr: io::stderr(),
            file,
        }
    }
}

impl Write for DualWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.stderr.write_all(bytes)?;
        if let Some(file) = &mut self.file {
            file.write_all(bytes)?;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stderr.flush()?;
        if let Some(file) = &mut self.file {
            file.flush()?;
        }
        Ok(())
    }
}
