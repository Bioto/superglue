//! Metrics helpers using the [`metrics`] facade.
//!
//! Call [`init_metrics`] once at startup (before emitting any metrics) to install
//! a recorder. The Prometheus exporter is available when the `prometheus` feature
//! is enabled; otherwise a no-op recorder is installed so the metric macros compile
//! and run without panicking.

// ---------------------------------------------------------------------------
// Metric name constants (stable API surface for operators)
// ---------------------------------------------------------------------------

/// Counter: total completion calls started (labels: model).
pub const COMPLETIONS_TOTAL: &str = "superglue.completions.total";
/// Counter: completion calls that returned an error (labels: model, error_kind).
pub const COMPLETIONS_ERRORS: &str = "superglue.completions.errors";
/// Counter: tool invocations dispatched (labels: tool_name).
pub const TOOL_CALLS_TOTAL: &str = "superglue.tool_calls.total";
/// Counter: tool invocations that failed (labels: tool_name, error_kind).
pub const TOOL_CALLS_ERRORS: &str = "superglue.tool_calls.errors";
/// Histogram: wall-clock ms per completion call (labels: model).
pub const COMPLETION_DURATION_MS: &str = "superglue.completion.duration_ms";
/// Counter: System One evaluations started (labels: model).
pub const SYSTEM_ONE_TOTAL: &str = "superglue.system_one.total";
/// Counter: System One evaluations that returned an error (labels: model, error_kind).
pub const SYSTEM_ONE_ERRORS: &str = "superglue.system_one.errors";
/// Histogram: wall-clock ms per System One evaluation (labels: model).
pub const SYSTEM_ONE_DURATION_MS: &str = "superglue.system_one.duration_ms";
/// Histogram: wall-clock ms per streaming completion (labels: model).
pub const STREAM_DURATION_MS: &str = "superglue.stream.duration_ms";
/// Histogram: number of items in a batch_complete call.
pub const BATCH_SIZE: &str = "superglue.batch.size";
/// Histogram: wall-clock ms for an entire batch_complete call.
pub const BATCH_DURATION_MS: &str = "superglue.batch.duration_ms";
/// Counter: number of HTTP retries performed.
pub const HTTP_RETRIES_TOTAL: &str = "superglue.http.retries.total";

// ---------------------------------------------------------------------------
// Recorder initialisation
// ---------------------------------------------------------------------------

#[cfg(feature = "prometheus")]
pub use metrics_exporter_prometheus::PrometheusHandle;

/// Install a Prometheus recorder and return a handle for on-demand rendering.
///
/// Embedders (e.g. harn) can expose `GET /metrics` on their own HTTP server using
/// [`PrometheusHandle::render`] instead of the standalone listener from [`init_metrics`].
#[cfg(feature = "prometheus")]
pub fn install_recorder() -> Result<PrometheusHandle, metrics_exporter_prometheus::BuildError> {
    metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder()
}

/// Install a metrics recorder.
///
/// - With the `prometheus` feature: installs a standalone scrape endpoint on
///   `0.0.0.0:9090/metrics` (see [`install_recorder`] for embedder-owned routes).
/// - Without: installs a no-op recorder so metric macros are harmless.
///
/// Call once at startup, before any metric macros run.
///
/// # Panics
///
/// Panics if the recorder has already been installed (call once per process).
pub fn init_metrics() {
    #[cfg(feature = "prometheus")]
    {
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .install()
            .expect("failed to install Prometheus metrics recorder");
    }

    #[cfg(not(feature = "prometheus"))]
    {
        // Install a no-op recorder so `metrics::counter!` etc. don't panic.
        let _ = metrics::set_global_recorder(NoopRecorder);
    }
}

/// Install a no-op or Prometheus recorder without starting a standalone HTTP listener.
///
/// Returns a [`PrometheusHandle`] when the `prometheus` feature is enabled so the
/// embedder can serve metrics from its own HTTP server.
#[cfg(feature = "prometheus")]
pub fn init_embedded_metrics() -> Option<PrometheusHandle> {
    install_recorder().ok()
}

#[cfg(not(feature = "prometheus"))]
pub fn init_embedded_metrics() -> Option<()> {
    let _ = metrics::set_global_recorder(NoopRecorder);
    None
}

// ---------------------------------------------------------------------------
// No-op recorder (non-prometheus builds)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "prometheus"))]
struct NoopRecorder;

#[cfg(not(feature = "prometheus"))]
impl metrics::Recorder for NoopRecorder {
    fn describe_counter(
        &self,
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
    }

    fn describe_gauge(
        &self,
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
    }

    fn describe_histogram(
        &self,
        _key: metrics::KeyName,
        _unit: Option<metrics::Unit>,
        _description: metrics::SharedString,
    ) {
    }

    fn register_counter(
        &self,
        _key: &metrics::Key,
        _metadata: &metrics::Metadata<'_>,
    ) -> metrics::Counter {
        metrics::Counter::noop()
    }

    fn register_gauge(
        &self,
        _key: &metrics::Key,
        _metadata: &metrics::Metadata<'_>,
    ) -> metrics::Gauge {
        metrics::Gauge::noop()
    }

    fn register_histogram(
        &self,
        _key: &metrics::Key,
        _metadata: &metrics::Metadata<'_>,
    ) -> metrics::Histogram {
        metrics::Histogram::noop()
    }
}
