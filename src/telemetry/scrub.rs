//! Field-scrubbing formatter for `tracing_subscriber::fmt`.
//!
//! [`ScrubFields`] is a [`tracing_subscriber::fmt::format::FormatFields`] implementation
//! that handles sensitive field names in formatted output according to a configurable
//! [`ScrubMode`]. It is used as the `N` type parameter of `tracing_subscriber::fmt::Layer<S, N>`.
//!
//! Fields in [`SENSITIVE_FIELDS`] are treated according to the mode:
//! - [`ScrubMode::Redact`] (default): value is replaced with `[REDACTED]`.
//! - [`ScrubMode::Hash`]: value is replaced with `[HASH:xxxxxxxx]` (8 hex digits of a
//!   `DefaultHasher` hash of the value's `Debug` representation). Safe for correlation
//!   across events without revealing content.
//! - [`ScrubMode::Allow`]: value is printed verbatim (development / local only).
//!
//! Non-sensitive fields always pass through unchanged.
//!
//! **Note**: this only scrubs the *fmt* (console / JSON) output layer. For OTLP spans the
//! sensitive attributes must be removed in a custom `SpanExporter` wrapper or Collector
//! processor.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use tracing::field::{Field, Visit};
use tracing_subscriber::field::RecordFields;
use tracing_subscriber::fmt::format::{FormatFields, Writer};

/// Field names whose values are treated as sensitive and handled via [`ScrubMode`].
pub const SENSITIVE_FIELDS: &[&str] = &[
    "api_key",
    "authorization",
    "content",
    "arguments",
    "result",
    "prompt",
    "system_prompt",
];

// ---------------------------------------------------------------------------
// ScrubMode
// ---------------------------------------------------------------------------

/// Controls how sensitive field values are handled in formatted log output.
///
/// Select a mode based on deployment context:
/// - **Production**: [`ScrubMode::Redact`] — no sensitive data in logs.
/// - **Debug sessions**: [`ScrubMode::Hash`] — values are hashed so you can
///   correlate identical values across events without revealing content.
/// - **Local development**: [`ScrubMode::Allow`] — plain values for maximum visibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScrubMode {
    /// Replace value with `[REDACTED]`. Safe for production. (default)
    #[default]
    Redact,
    /// Replace value with `[HASH:xxxxxxxx]` — 8 hex digits of a `DefaultHasher`
    /// hash of the value's `Debug` representation. Allows event correlation
    /// without exposing the raw value.
    Hash,
    /// Print field values unchanged. Only for local development.
    Allow,
}

impl ScrubMode {
    /// Parse from a string (`"redact"`, `"hash"`, `"allow"`).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "redact" => Some(Self::Redact),
            "hash" => Some(Self::Hash),
            "allow" => Some(Self::Allow),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Public formatter type
// ---------------------------------------------------------------------------

/// A `FormatFields` implementation that handles [`SENSITIVE_FIELDS`] via [`ScrubMode`].
///
/// Use it when building the `fmt` subscriber layer:
/// ```rust,no_run
/// # use superglue::telemetry::scrub::{ScrubFields, ScrubMode};
/// # use tracing_subscriber::Layer;
/// let layer = tracing_subscriber::fmt::layer::<tracing_subscriber::Registry>()
///     .fmt_fields(ScrubFields(ScrubMode::Redact));
/// ```
pub struct ScrubFields(pub ScrubMode);

impl Default for ScrubFields {
    fn default() -> Self {
        ScrubFields(ScrubMode::Redact)
    }
}

impl<'writer> FormatFields<'writer> for ScrubFields {
    fn format_fields<R: RecordFields>(
        &self,
        mut writer: Writer<'writer>,
        fields: R,
    ) -> fmt::Result {
        let mut visitor = ScrubVisitor::new(&mut writer, self.0);
        fields.record(&mut visitor);
        visitor.finish()
    }
}

// ---------------------------------------------------------------------------
// Internal visitor
// ---------------------------------------------------------------------------

fn scrub_value(name: &str, value: &dyn fmt::Debug, mode: ScrubMode) -> String {
    if !SENSITIVE_FIELDS.contains(&name) {
        return format!("{value:?}");
    }
    match mode {
        ScrubMode::Redact => "[REDACTED]".to_string(),
        ScrubMode::Hash => {
            let mut h = DefaultHasher::new();
            format!("{value:?}").hash(&mut h);
            // Truncate to lower 32 bits → 8 hex digits for a compact representation.
            format!("[HASH:{:08x}]", h.finish() as u32)
        }
        ScrubMode::Allow => format!("{value:?}"),
    }
}

struct ScrubVisitor<'a, 'w> {
    writer: &'a mut Writer<'w>,
    mode: ScrubMode,
    is_first: bool,
    result: fmt::Result,
}

impl<'a, 'w> ScrubVisitor<'a, 'w> {
    fn new(writer: &'a mut Writer<'w>, mode: ScrubMode) -> Self {
        ScrubVisitor {
            writer,
            mode,
            is_first: true,
            result: Ok(()),
        }
    }

    fn finish(self) -> fmt::Result {
        self.result
    }

    fn write_separator(&mut self) {
        if !self.is_first {
            self.result = write!(self.writer, ", ");
        }
        self.is_first = false;
    }

    fn write_field(&mut self, name: &str, value: &dyn fmt::Debug) {
        self.write_separator();
        if self.result.is_ok() {
            let rendered = scrub_value(name, value, self.mode);
            self.result = write!(self.writer, "{name}={rendered}");
        }
    }
}

/// JSON-object field formatter with the same scrubbing rules as [`ScrubFields`].
///
/// The tracing JSON layer expects span field blobs to be JSON objects (see
/// `tracing_subscriber::fmt::format::Json`). Pretty `key=value` output panics
/// that layer in debug builds — use this type with `.json()`.
pub struct ScrubJsonFields(pub ScrubMode);

impl Default for ScrubJsonFields {
    fn default() -> Self {
        ScrubJsonFields(ScrubMode::Redact)
    }
}

impl<'writer> FormatFields<'writer> for ScrubJsonFields {
    fn format_fields<R: RecordFields>(
        &self,
        mut writer: Writer<'writer>,
        fields: R,
    ) -> fmt::Result {
        let mut values = std::collections::BTreeMap::<String, serde_json::Value>::new();
        {
            let mut visitor = ScrubJsonVisitor {
                values: &mut values,
                mode: self.0,
            };
            fields.record(&mut visitor);
        }
        let json = serde_json::to_string(&values).map_err(|_| fmt::Error)?;
        write!(writer, "{json}")
    }

    fn add_fields(
        &self,
        current: &'writer mut tracing_subscriber::fmt::FormattedFields<Self>,
        fields: &tracing::span::Record<'_>,
    ) -> fmt::Result {
        let mut values: std::collections::BTreeMap<String, serde_json::Value> =
            if current.is_empty() {
                std::collections::BTreeMap::new()
            } else {
                serde_json::from_str(&current.fields).unwrap_or_default()
            };
        {
            let mut visitor = ScrubJsonVisitor {
                values: &mut values,
                mode: self.0,
            };
            fields.record(&mut visitor);
        }
        current.fields = serde_json::to_string(&values).map_err(|_| fmt::Error)?;
        Ok(())
    }
}

fn scrub_json_value(name: &str, value: serde_json::Value, mode: ScrubMode) -> serde_json::Value {
    if !SENSITIVE_FIELDS.contains(&name) {
        return value;
    }
    match mode {
        ScrubMode::Redact => serde_json::Value::String("[REDACTED]".into()),
        ScrubMode::Hash => {
            let mut h = DefaultHasher::new();
            value.to_string().hash(&mut h);
            serde_json::Value::String(format!("[HASH:{:08x}]", h.finish() as u32))
        }
        ScrubMode::Allow => value,
    }
}

struct ScrubJsonVisitor<'a> {
    values: &'a mut std::collections::BTreeMap<String, serde_json::Value>,
    mode: ScrubMode,
}

impl Visit for ScrubJsonVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let name = field.name();
        let rendered = scrub_value(name, value, self.mode);
        // Debug strings often include quotes; store as a plain JSON string without
        // the Debug wrapping when the scrubber already returned a display string.
        let json_value = if SENSITIVE_FIELDS.contains(&name) {
            serde_json::Value::String(rendered)
        } else {
            serde_json::Value::String(format!("{value:?}"))
        };
        self.values.insert(name.to_string(), json_value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let name = field.name();
        self.values.insert(
            name.to_string(),
            scrub_json_value(name, serde_json::Value::String(value.to_string()), self.mode),
        );
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        let name = field.name();
        self.values.insert(
            name.to_string(),
            scrub_json_value(name, serde_json::Value::from(value), self.mode),
        );
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        let name = field.name();
        self.values.insert(
            name.to_string(),
            scrub_json_value(name, serde_json::Value::from(value), self.mode),
        );
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        let name = field.name();
        self.values.insert(
            name.to_string(),
            scrub_json_value(name, serde_json::Value::from(value), self.mode),
        );
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        let name = field.name();
        let json = serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or_else(|| serde_json::Value::String(value.to_string()));
        self.values
            .insert(name.to_string(), scrub_json_value(name, json, self.mode));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.values.insert(
            field.name().to_string(),
            serde_json::Value::String(value.to_string()),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_fields_cover_required_names() {
        for name in &["api_key", "authorization", "content", "arguments", "result"] {
            assert!(
                SENSITIVE_FIELDS.contains(name),
                "{name} should be in SENSITIVE_FIELDS"
            );
        }
    }

    #[test]
    fn safe_fields_not_in_deny_list() {
        for name in &["model", "request_id", "rounds", "tool_name", "elapsed_ms"] {
            assert!(
                !SENSITIVE_FIELDS.contains(name),
                "{name} should NOT be in SENSITIVE_FIELDS"
            );
        }
    }

    #[test]
    fn scrub_mode_redact_replaces_sensitive_value() {
        let rendered = scrub_value("api_key", &"secret-value", ScrubMode::Redact);
        assert_eq!(rendered, "[REDACTED]");
    }

    #[test]
    fn scrub_mode_hash_produces_hash_prefix() {
        let rendered = scrub_value("api_key", &"secret-value", ScrubMode::Hash);
        assert!(
            rendered.starts_with("[HASH:"),
            "expected [HASH:...], got: {rendered}"
        );
        assert!(
            rendered.ends_with(']'),
            "expected closing ], got: {rendered}"
        );
        assert_eq!(
            rendered.len(),
            "[HASH:xxxxxxxx]".len(),
            "hash should be 8 hex digits"
        );
    }

    #[test]
    fn scrub_mode_hash_is_deterministic() {
        let a = scrub_value("content", &"hello", ScrubMode::Hash);
        let b = scrub_value("content", &"hello", ScrubMode::Hash);
        assert_eq!(a, b, "same value must produce same hash");
    }

    #[test]
    fn scrub_mode_hash_distinguishes_different_values() {
        let a = scrub_value("content", &"hello", ScrubMode::Hash);
        let b = scrub_value("content", &"world", ScrubMode::Hash);
        assert_ne!(
            a, b,
            "different values should (likely) produce different hashes"
        );
    }

    #[test]
    fn scrub_mode_allow_passes_through_sensitive() {
        let rendered = scrub_value("api_key", &"my-secret", ScrubMode::Allow);
        assert!(
            rendered.contains("my-secret"),
            "Allow mode should not redact: {rendered}"
        );
    }

    #[test]
    fn scrub_mode_non_sensitive_always_passes_through() {
        for mode in [ScrubMode::Redact, ScrubMode::Hash, ScrubMode::Allow] {
            let rendered = scrub_value("model", &"gpt-5.4-nano-2026-03-17", mode);
            assert!(
                rendered.contains("gpt-5.4-nano-2026-03-17"),
                "non-sensitive field should pass through in {mode:?}: {rendered}"
            );
        }
    }

    #[test]
    fn scrub_mode_from_str_parses_all_variants() {
        assert_eq!(ScrubMode::from_str("redact"), Some(ScrubMode::Redact));
        assert_eq!(ScrubMode::from_str("REDACT"), Some(ScrubMode::Redact));
        assert_eq!(ScrubMode::from_str("hash"), Some(ScrubMode::Hash));
        assert_eq!(ScrubMode::from_str("allow"), Some(ScrubMode::Allow));
        assert_eq!(ScrubMode::from_str("unknown"), None);
    }

    #[test]
    fn scrub_fields_default_is_redact() {
        let sf = ScrubFields::default();
        assert_eq!(sf.0, ScrubMode::Redact);
    }

    #[test]
    fn scrub_fields_is_non_empty() {
        assert!(!SENSITIVE_FIELDS.is_empty());
    }

    #[test]
    fn scrub_json_fields_emits_parseable_object() {
        let mut values = std::collections::BTreeMap::<String, serde_json::Value>::new();
        values.insert(
            "model".into(),
            scrub_json_value(
                "model",
                serde_json::Value::String("gpt-5.6-luna".into()),
                ScrubMode::Redact,
            ),
        );
        values.insert(
            "api_key".into(),
            scrub_json_value(
                "api_key",
                serde_json::Value::String("secret".into()),
                ScrubMode::Redact,
            ),
        );
        let json = serde_json::to_string(&values).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
        assert_eq!(parsed["model"], "gpt-5.6-luna");
        assert_eq!(parsed["api_key"], "[REDACTED]");
    }

    #[test]
    fn scrub_json_empty_is_object() {
        let values = std::collections::BTreeMap::<String, serde_json::Value>::new();
        let json = serde_json::to_string(&values).unwrap();
        assert_eq!(json, "{}");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
    }
}

impl<'a, 'w> Visit for ScrubVisitor<'a, 'w> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.write_field(field.name(), value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        // Route through write_field so ScrubMode applies uniformly.
        self.write_field(field.name(), &value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.write_field(field.name(), &value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.write_field(field.name(), &value);
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.write_field(field.name(), &value);
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.write_field(field.name(), &value);
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.write_separator();
        if self.result.is_ok() {
            // Errors are not scrubbed — they are operational signals without payload data.
            self.result = write!(self.writer, "{}={value}", field.name());
        }
    }
}
