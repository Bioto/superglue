//! TypeSafe-backed guardrail. Fail-closed on HTTP or parse errors.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::guardrails::{GuardrailHandler, GuardrailOutcome, GuardrailStage};
use crate::http::HttpClient;
use crate::providers::ProviderCredentials;

use super::client::system_one;
use super::helpers::{choice_if_confident, noul_yes, score_at_least};
use super::types::{Questions, State, SystemOneRequest, SystemOneResponse};

/// Shared fail-closed reason. Do not include upstream bodies.
const UNAVAILABLE: &str = "typesafe guardrail unavailable";

/// A single block condition over a named System One answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypeSafeGuardrailRule {
    #[serde(rename = "noul_yes")]
    BlockIfNoulYes { question_id: String, threshold: f64 },
    #[serde(rename = "choice")]
    BlockIfChoice {
        question_id: String,
        values: Vec<String>,
        #[serde(default)]
        min_confidence: f64,
    },
    #[serde(rename = "score_at_least")]
    BlockIfScoreAtLeast { question_id: String, level: f64 },
}

/// Caller-supplied questions, model, and block rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSafeGuardrailPolicy {
    #[serde(default = "default_policy_model")]
    pub model: String,
    pub questions: Questions,
    #[serde(default)]
    pub rules: Vec<TypeSafeGuardrailRule>,
}

fn default_policy_model() -> String {
    super::types::DEFAULT_MODEL.to_string()
}

impl TypeSafeGuardrailPolicy {
    #[must_use]
    pub fn new(questions: Questions) -> Self {
        Self {
            model: super::types::DEFAULT_MODEL.to_string(),
            questions,
            rules: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    #[must_use]
    pub fn with_rule(mut self, rule: TypeSafeGuardrailRule) -> Self {
        self.rules.push(rule);
        self
    }
}

/// Evaluates message text with TypeSafe System One.
pub struct TypeSafeGuardrail {
    http: Arc<HttpClient>,
    credentials: ProviderCredentials,
    policy: TypeSafeGuardrailPolicy,
}

impl TypeSafeGuardrail {
    #[must_use]
    pub fn new(
        http: Arc<HttpClient>,
        credentials: ProviderCredentials,
        policy: TypeSafeGuardrailPolicy,
    ) -> Self {
        Self {
            http,
            credentials,
            policy,
        }
    }
}

#[async_trait]
impl GuardrailHandler for TypeSafeGuardrail {
    async fn check(&self, _stage: GuardrailStage, content: &str) -> GuardrailOutcome {
        if self.policy.questions.0.is_empty() {
            return GuardrailOutcome::Allow(content.to_string());
        }
        let request = SystemOneRequest::new(State::from(content), self.policy.questions.clone())
            .with_model(self.policy.model.clone());
        let response = match system_one(&self.http, &self.credentials, request).await {
            Ok(response) => response,
            Err(_) => return GuardrailOutcome::Block(UNAVAILABLE.to_string()),
        };

        for rule in &self.policy.rules {
            if let Some(reason) = rule_blocks(rule, &response) {
                return GuardrailOutcome::Block(reason);
            }
        }
        GuardrailOutcome::Allow(content.to_string())
    }
}

fn rule_blocks(rule: &TypeSafeGuardrailRule, response: &SystemOneResponse) -> Option<String> {
    match rule {
        TypeSafeGuardrailRule::BlockIfNoulYes {
            question_id,
            threshold,
        } => {
            let answer = response.answer(question_id)?;
            noul_yes(answer, *threshold)
                .then(|| format!("typesafe noul {question_id} exceeded threshold"))
        }
        TypeSafeGuardrailRule::BlockIfChoice {
            question_id,
            values,
            min_confidence,
        } => {
            let answer = response.answer(question_id)?;
            choice_if_confident(answer, *min_confidence).and_then(|choice| {
                values
                    .iter()
                    .any(|blocked| blocked == choice)
                    .then(|| format!("typesafe choice {question_id} matched a blocked option"))
            })
        }
        TypeSafeGuardrailRule::BlockIfScoreAtLeast { question_id, level } => {
            let answer = response.answer(question_id)?;
            score_at_least(answer, *level)
                .then(|| format!("typesafe score {question_id} exceeded level"))
        }
    }
}
