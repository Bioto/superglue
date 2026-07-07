//! Gateway HTTP error types.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("{message}")]
    WithStatus {
        status: StatusCode,
        kind: String,
        message: String,
    },
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("internal error: {0}")]
    Internal(String),
}

impl GatewayError {
    #[must_use]
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::UNAUTHORIZED,
            kind: "authentication_error".into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::FORBIDDEN,
            kind: "permission_error".into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error".into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::NOT_FOUND,
            kind: "not_found".into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn budget_exceeded(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "budget_exceeded".into(),
            message: message.into(),
        }
    }

    #[must_use]
    pub fn upstream(message: impl Into<String>) -> Self {
        Self::WithStatus {
            status: StatusCode::BAD_GATEWAY,
            kind: "upstream_error".into(),
            message: message.into(),
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, kind, message) = match self {
            GatewayError::WithStatus {
                status,
                kind,
                message,
            } => (status, kind, message),
            GatewayError::Db(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "database_error".into(),
                e.to_string(),
            ),
            GatewayError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error".into(),
                msg,
            ),
        };
        let body = ErrorBody {
            error: ErrorDetail { message, kind },
        };
        (status, axum::Json(body)).into_response()
    }
}

pub type GatewayResult<T> = Result<T, GatewayError>;
