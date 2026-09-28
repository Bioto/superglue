//! Shared benchmark config matrix for context optimization examples and wiremock tests.

use superglue::tools::ToolMode;

use super::conversation_profiles::ConversationProfile;

/// One row in the comparison benchmark matrix.
pub struct BenchConfig {
    pub label: &'static str,
    pub tool_mode: ToolMode,
    pub condense_tool_messages: bool,
    pub aaak_tool_condensing: bool,
}

/// High-level grouping for benchmark reporting semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKind {
    Standard,
    Dynamic,
    CondensePlain,
    CondenseAaak,
    Combined,
    Code,
}

impl BenchConfig {
    #[must_use]
    pub const fn kind(&self) -> ConfigKind {
        match (
            self.tool_mode,
            self.condense_tool_messages,
            self.aaak_tool_condensing,
        ) {
            (ToolMode::Standard, false, _) => ConfigKind::Standard,
            (ToolMode::Dynamic, false, _) => ConfigKind::Dynamic,
            (ToolMode::Code, false, _) => ConfigKind::Code,
            (ToolMode::Standard, true, false) => ConfigKind::CondensePlain,
            (ToolMode::Standard, true, true) => ConfigKind::CondenseAaak,
            (ToolMode::Dynamic, true, true) => ConfigKind::Combined,
            (ToolMode::Dynamic, true, false) => ConfigKind::Dynamic,
            (ToolMode::Code, true, _) => ConfigKind::Code,
        }
    }

    /// Expected max tools array length per LLM request (9-tool fixture).
    #[must_use]
    pub fn expected_peak_tools(&self, profile: ConversationProfile) -> usize {
        match self.tool_mode {
            ToolMode::Standard => 9,
            ToolMode::Dynamic => profile.expected_tools().len() + 1,
            ToolMode::Code => 2,
        }
    }
}

pub const BENCH_CONFIGS: &[BenchConfig] = &[
    BenchConfig {
        label: "standard",
        tool_mode: ToolMode::Standard,
        condense_tool_messages: false,
        aaak_tool_condensing: false,
    },
    BenchConfig {
        label: "dynamic",
        tool_mode: ToolMode::Dynamic,
        condense_tool_messages: false,
        aaak_tool_condensing: false,
    },
    BenchConfig {
        label: "condense_plain",
        tool_mode: ToolMode::Standard,
        condense_tool_messages: true,
        aaak_tool_condensing: false,
    },
    BenchConfig {
        label: "condense_aaak",
        tool_mode: ToolMode::Standard,
        condense_tool_messages: true,
        aaak_tool_condensing: true,
    },
    BenchConfig {
        label: "dynamic_condense_aaak",
        tool_mode: ToolMode::Dynamic,
        condense_tool_messages: true,
        aaak_tool_condensing: true,
    },
    BenchConfig {
        label: "code",
        tool_mode: ToolMode::Code,
        condense_tool_messages: false,
        aaak_tool_condensing: false,
    },
    BenchConfig {
        label: "code_condense",
        tool_mode: ToolMode::Code,
        condense_tool_messages: true,
        aaak_tool_condensing: false,
    },
];

/// Percent delta vs baseline: `-N%` when smaller (savings), `+N%` when larger.
#[must_use]
pub fn format_token_delta(baseline: u32, value: u32) -> String {
    if baseline == 0 {
        return "—".to_string();
    }
    if value == baseline {
        return "0%".to_string();
    }
    let pct = (value as f64 / baseline as f64 - 1.0) * 100.0;
    if pct > 0.0 {
        format!("+{pct:.0}%")
    } else {
        format!("{pct:.0}%")
    }
}

/// Cumulative total-token savings vs standard baseline.
#[must_use]
pub fn token_savings_label(kind: ConfigKind, baseline: u32, value: u32) -> String {
    match kind {
        ConfigKind::Standard => "—".to_string(),
        _ => format_token_delta(baseline, value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn savings_shows_minus_percent() {
        assert_eq!(format_token_delta(1000, 800), "-20%");
    }

    #[test]
    fn larger_tokens_shows_plus_percent() {
        assert_eq!(format_token_delta(1000, 1200), "+20%");
    }
}
