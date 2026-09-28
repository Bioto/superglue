//! TypeSafe System One evaluation client, helpers, and guardrail.

mod client;
mod guardrail;
mod helpers;
mod types;

pub use client::{
    SYSTEM_ONE_PATH, SystemOneError, qualify_model, request_from_parts, system_one,
    wire_system_one_model,
};
pub use guardrail::{TypeSafeGuardrail, TypeSafeGuardrailPolicy, TypeSafeGuardrailRule};
pub use helpers::{STATE_SNIPPET_CHARS, choice_if_confident, clip_state, noul_yes, score_at_least};
pub use types::{
    Answer, Choice, DEFAULT_MODEL, Noul, NoulCriteria, Question, Questions, Score, State,
    SystemOneRequest, SystemOneResponse, SystemOneUsage,
};

#[cfg(test)]
mod serde_tests {
    use super::*;

    #[test]
    fn serializes_mixed_questions() {
        let mut questions = Questions::new();
        questions.insert(
            "is_urgent",
            Noul::new("Does this message express urgency?")
                .criteria("Explicitly time-sensitive", "No urgency expressed"),
        );
        questions.insert(
            "department",
            Choice::new("Which team should handle this")
                .option("billing", "Payment or subscription issues")
                .option("technical", "Bugs or integration problems"),
        );
        questions.insert(
            "frustration",
            Score::new("How frustrated the customer appears").levels([
                "Calm",
                "Frustrated",
                "Very angry",
            ]),
        );
        let request = SystemOneRequest::new("Help ASAP", questions).with_model("jev-latest");
        let json = serde_json::to_value(&request).expect("serialize");
        assert_eq!(json["questions"]["is_urgent"]["type"], "noul");
        assert_eq!(json["questions"]["department"]["type"], "choice");
        assert_eq!(json["questions"]["frustration"]["type"], "score");
        assert_eq!(
            json["questions"]["frustration"]["criteria"]
                .as_array()
                .expect("levels")
                .len(),
            3
        );
    }

    #[test]
    fn deserializes_sample_answers() {
        let json = serde_json::json!({
            "model": "jev-latest",
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": { "billing": 0.159, "technical": 0.84, "sales": 0.001 },
                    "confidence": 0.596
                },
                "frustration": {
                    "type": "score",
                    "score": 1.035,
                    "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                    "confidence": 0.842
                },
                "is_urgent": { "type": "noul", "noul": 0.999 }
            },
            "usage": { "input_tokens": 312, "output_tokens": 48 }
        });
        let response: SystemOneResponse = serde_json::from_value(json).expect("parse");
        assert_eq!(
            response.answer("department").and_then(Answer::choice),
            Some("technical")
        );
        assert_eq!(
            response.answer("is_urgent").and_then(Answer::noul),
            Some(0.999)
        );
        assert!(
            (response
                .answer("frustration")
                .and_then(Answer::score)
                .unwrap()
                - 1.035)
                .abs()
                < f64::EPSILON
        );
        assert_eq!(response.usage.input_tokens, Some(312));
        let proto = response.usage.to_proto();
        assert_eq!(proto.prompt_tokens, 312);
        assert_eq!(proto.completion_tokens, 48);
    }
}
