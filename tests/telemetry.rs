//! Smoke tests for superglue::telemetry — verifies init_tracing and metrics initialisation
//! are callable without panicking.

use superglue::telemetry::{ScrubMode, TelemetryConfig, init_tracing};
use superglue::telemetry::metrics::{
    BATCH_DURATION_MS, BATCH_SIZE, COMPLETION_DURATION_MS, COMPLETIONS_ERRORS,
    COMPLETIONS_TOTAL, HTTP_RETRIES_TOTAL, STREAM_DURATION_MS, TOOL_CALLS_ERRORS,
    TOOL_CALLS_TOTAL,
};

#[test]
fn init_tracing_is_idempotent() {
    // Call twice — the second call should be a no-op (try_init returns Err but
    // init_tracing swallows it).
    init_tracing(TelemetryConfig::default());
    init_tracing(TelemetryConfig {
        log_level: "warn".to_string(),
        scrub_mode: ScrubMode::Redact,
        otlp_endpoint: None,
    });
}

#[test]
fn init_tracing_hash_mode_is_idempotent() {
    init_tracing(TelemetryConfig {
        scrub_mode: ScrubMode::Hash,
        log_level: "error".to_string(),
        otlp_endpoint: None,
    });
}

#[test]
fn init_tracing_allow_mode_is_idempotent() {
    init_tracing(TelemetryConfig {
        scrub_mode: ScrubMode::Allow,
        log_level: "error".to_string(),
        otlp_endpoint: None,
    });
}

#[test]
fn telemetry_config_default_values() {
    let cfg = TelemetryConfig::default();
    assert_eq!(cfg.log_level, "info");
    assert_eq!(cfg.scrub_mode, ScrubMode::Redact);
    assert!(cfg.otlp_endpoint.is_none());
}

#[test]
fn metric_name_constants_are_non_empty() {
    // Verify the stable metric name constants haven't been accidentally emptied.
    assert!(!COMPLETIONS_TOTAL.is_empty());
    assert!(!COMPLETIONS_ERRORS.is_empty());
    assert!(!TOOL_CALLS_TOTAL.is_empty());
    assert!(!TOOL_CALLS_ERRORS.is_empty());
    assert!(!COMPLETION_DURATION_MS.is_empty());
    assert!(!STREAM_DURATION_MS.is_empty());
    assert!(!BATCH_SIZE.is_empty());
    assert!(!BATCH_DURATION_MS.is_empty());
    assert!(!HTTP_RETRIES_TOTAL.is_empty());
}

#[test]
fn metric_names_have_superglue_prefix() {
    for name in &[
        COMPLETIONS_TOTAL,
        COMPLETIONS_ERRORS,
        TOOL_CALLS_TOTAL,
        TOOL_CALLS_ERRORS,
        COMPLETION_DURATION_MS,
        STREAM_DURATION_MS,
        BATCH_SIZE,
        BATCH_DURATION_MS,
    ] {
        assert!(
            name.starts_with("superglue."),
            "{name} should start with 'superglue.'"
        );
    }
}
