//! TypeSafe System One request and response types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::proto;
use crate::usage::{UsageBreakdown, usage_from_breakdown};

/// Default TypeSafe System One model.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Evaluation input: plain text or structured JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum State {
    Text(String),
    Json(Value),
}

impl From<&str> for State {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for State {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

/// Optional yes/no descriptions for a Noul question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

/// A typed System One question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        instructions: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        instructions: String,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

/// Builder for a yes/no Noul question.
#[derive(Debug, Clone, PartialEq)]
pub struct Noul {
    instructions: String,
    criteria: Option<NoulCriteria>,
}

impl Noul {
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            criteria: None,
        }
    }

    #[must_use]
    pub fn criteria(mut self, yes: impl Into<String>, no: impl Into<String>) -> Self {
        self.criteria = Some(NoulCriteria {
            yes: Some(yes.into()),
            no: Some(no.into()),
        });
        self
    }

    #[must_use]
    pub fn into_question(self) -> Question {
        Question::Noul {
            instructions: self.instructions,
            criteria: self.criteria,
        }
    }
}

impl From<Noul> for Question {
    fn from(value: Noul) -> Self {
        value.into_question()
    }
}

/// Builder for a closed-set Choice question.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    instructions: String,
    criteria: BTreeMap<String, Value>,
}

impl Choice {
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            criteria: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn option(mut self, key: impl Into<String>, description: impl Into<String>) -> Self {
        self.criteria
            .insert(key.into(), Value::String(description.into()));
        self
    }

    #[must_use]
    pub fn options(mut self, criteria: BTreeMap<String, Value>) -> Self {
        self.criteria = criteria;
        self
    }

    #[must_use]
    pub fn into_question(self) -> Question {
        Question::Choice {
            instructions: self.instructions,
            criteria: self.criteria,
        }
    }
}

impl From<Choice> for Question {
    fn from(value: Choice) -> Self {
        value.into_question()
    }
}

/// Builder for an ordered Score question.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    instructions: String,
    criteria: Vec<String>,
}

impl Score {
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            criteria: Vec::new(),
        }
    }

    #[must_use]
    pub fn levels(mut self, criteria: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.criteria = criteria.into_iter().map(Into::into).collect();
        self
    }

    #[must_use]
    pub fn into_question(self) -> Question {
        Question::Score {
            instructions: self.instructions,
            criteria: self.criteria,
        }
    }
}

impl From<Score> for Question {
    fn from(value: Score) -> Self {
        value.into_question()
    }
}

/// Named questions sent in one System One call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Questions(pub BTreeMap<String, Question>);

impl Questions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: impl Into<String>, question: impl Into<Question>) -> &mut Self {
        self.0.insert(id.into(), question.into());
        self
    }
}

impl From<BTreeMap<String, Question>> for Questions {
    fn from(value: BTreeMap<String, Question>) -> Self {
        Self(value)
    }
}

/// A typed System One answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
    Score {
        score: f64,
        #[serde(default)]
        legend: BTreeMap<String, Value>,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        confidence: f64,
    },
}

impl Answer {
    #[must_use]
    pub fn noul(&self) -> Option<f64> {
        match self {
            Self::Noul { noul } => Some(*noul),
            _ => None,
        }
    }

    #[must_use]
    pub fn choice(&self) -> Option<&str> {
        match self {
            Self::Choice { choice, .. } => Some(choice.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn score(&self) -> Option<f64> {
        match self {
            Self::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    #[must_use]
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => Some(*confidence),
            Self::Noul { .. } => None,
        }
    }

    #[must_use]
    pub fn probabilities(&self) -> Option<&BTreeMap<String, f64>> {
        match self {
            Self::Choice { probabilities, .. } | Self::Score { probabilities, .. } => {
                Some(probabilities)
            }
            Self::Noul { .. } => None,
        }
    }
}

/// Token counts from a System One response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SystemOneUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

impl SystemOneUsage {
    #[must_use]
    pub fn to_proto(&self) -> proto::Usage {
        usage_from_breakdown(UsageBreakdown::from_counts(
            self.input_tokens.unwrap_or(0),
            self.output_tokens.unwrap_or(0),
        ))
    }
}

/// Request body for `POST /v1/systemone`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub questions: BTreeMap<String, Question>,
}

impl SystemOneRequest {
    #[must_use]
    pub fn new(state: impl Into<State>, questions: Questions) -> Self {
        Self {
            state: state.into(),
            model: None,
            questions: questions.0,
        }
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Response body from `POST /v1/systemone`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub usage: SystemOneUsage,
}

impl SystemOneResponse {
    #[must_use]
    pub fn answer(&self, id: &str) -> Option<&Answer> {
        self.answers.get(id)
    }
}
