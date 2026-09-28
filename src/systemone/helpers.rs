//! Decision helpers for System One answers.

use super::types::Answer;

/// Default character budget for untrusted text sent as System One state.
pub const STATE_SNIPPET_CHARS: usize = 240;

/// Truncate untrusted text before it becomes System One state.
#[must_use]
pub fn clip_state(text: &str, max: usize) -> String {
    let max = if max == 0 { STATE_SNIPPET_CHARS } else { max };
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= max {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

/// Return true when a Noul answer is at or above `threshold`.
#[must_use]
pub fn noul_yes(answer: &Answer, threshold: f64) -> bool {
    answer.noul().is_some_and(|value| value >= threshold)
}

/// Return the selected choice when confidence is at or above `min_confidence`.
#[must_use]
pub fn choice_if_confident(answer: &Answer, min_confidence: f64) -> Option<&str> {
    match answer {
        Answer::Choice {
            choice, confidence, ..
        } if *confidence >= min_confidence => Some(choice.as_str()),
        _ => None,
    }
}

/// Return true when a Score answer is at or above `level`.
#[must_use]
pub fn score_at_least(answer: &Answer, level: f64) -> bool {
    answer.score().is_some_and(|value| value >= level)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn noul_yes_respects_threshold() {
        let yes = Answer::Noul { noul: 0.8 };
        let no = Answer::Noul { noul: 0.2 };
        assert!(noul_yes(&yes, 0.7));
        assert!(!noul_yes(&no, 0.7));
        assert!(!noul_yes(
            &Answer::Choice {
                choice: "a".into(),
                probabilities: BTreeMap::new(),
                confidence: 1.0,
            },
            0.5
        ));
    }

    #[test]
    fn choice_if_confident_requires_confidence() {
        let answer = Answer::Choice {
            choice: "technical".into(),
            probabilities: BTreeMap::new(),
            confidence: 0.6,
        };
        assert_eq!(choice_if_confident(&answer, 0.5), Some("technical"));
        assert_eq!(choice_if_confident(&answer, 0.7), None);
    }

    #[test]
    fn clip_state_adds_ellipsis() {
        assert_eq!(clip_state("abcd", 3), "abc…");
        assert_eq!(clip_state("ab", 3), "ab");
    }

    #[test]
    fn score_at_least_compares_level() {
        let answer = Answer::Score {
            score: 1.4,
            legend: BTreeMap::new(),
            probabilities: BTreeMap::new(),
            confidence: 0.8,
        };
        assert!(score_at_least(&answer, 1.0));
        assert!(!score_at_least(&answer, 1.5));
    }
}
