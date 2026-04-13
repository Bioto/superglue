//! Metrics helpers using the [`metrics`] facade.
//!
//! Call [`init_metrics`] once at startup (before emitting any metrics) to install
//! a recorder. The Prometheus exporter is available when the `prometheus` feature
//! is enabled; otherwise a no-op recorder is installed so the metric macros compile
//! and run without panicking.
//!
//! Emit metrics at the call sites with the standard `metrics` macros:
//!
//! ```rust,no_run
//! metrics::counter!("superglue.completions.total").increment(1);
//! metrics::histogram!("superglue.completion.duration_ms").record(42.0);
//! ```

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

/// Install a metrics recorder.
///
/// - With the `prometheus` feature: installs a `PrometheusBuilder` recorder that
///   serves metrics on `0.0.0.0:9090/metrics` by default.
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
