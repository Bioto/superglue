#![cfg(feature = "gateway")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::gateway::{router, test_state, test_state_with_openai};

const MASTER_KEY: &str = "test-master-key-12345";

async fn body_to_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(json!({ "raw": String::from_utf8_lossy(&bytes) }))
}

fn auth_request(
    method: &str,
    uri: &str,
    key: &str,
    body: Option<serde_json::Value>,
) -> Request<Body> {
    let mut builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("X-Superglue-Key", format!("Bearer {key}"));
    if let Some(b) = body {
        builder = builder.header("content-type", "application/json");
        builder
            .body(Body::from(serde_json::to_vec(&b).unwrap()))
            .unwrap()
    } else {
        builder.body(Body::empty()).unwrap()
    }
}

#[tokio::test]
async fn virtual_key_allowed_model_proxies_and_logs_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        })))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let state = test_state_with_openai(
        MASTER_KEY,
        &dir.path().join("gw.db"),
        &server.uri(),
        "sk-test",
    );
    let app = router(state.clone());

    // Setup user + key via admin
    let setup = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "user-1", "alias": "Alice" })),
        ))
        .await
        .unwrap();
    assert_eq!(setup.status(), StatusCode::OK);

    let key_resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/keys",
            MASTER_KEY,
            Some(json!({
                "name": "test-key",
                "user_id": "user-1",
                "allowed_models": ["openai:gpt-4o-mini"]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(key_resp.status(), StatusCode::OK);
    let key_json = body_to_json(key_resp.into_body()).await;
    let virtual_key = key_json["key"].as_str().unwrap();

    // Completion via virtual key
    let resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "hello" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let usage = state.db.list_usage(None, None, 10).unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].user_id, "user-1");
    assert_eq!(usage[0].prompt_tokens, 10);
}

#[tokio::test]
async fn virtual_key_disallowed_model_returns_403() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state.clone());

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "user-1" })),
        ))
        .await
        .unwrap();

    let key_resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/keys",
            MASTER_KEY,
            Some(json!({
                "user_id": "user-1",
                "allowed_models": ["openai:gpt-4o-mini"]
            })),
        ))
        .await
        .unwrap();
    let virtual_key = body_to_json(key_resp.into_body()).await["key"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = app
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o",
                "messages": [{ "role": "user", "content": "hello" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn virtual_key_creation_requires_user_id() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state);

    let resp = app
        .oneshot(auth_request(
            "POST",
            "/v1/keys",
            MASTER_KEY,
            Some(json!({
                "user_id": "missing-user",
                "allowed_models": ["openai:*"]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn master_key_requires_user_field() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    let state = test_state_with_openai(
        MASTER_KEY,
        &dir.path().join("gw.db"),
        &server.uri(),
        "sk-test",
    );
    let app = router(state);

    let resp = app
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            MASTER_KEY,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "hello" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn budget_enforce_returns_429() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state.clone());

    let budget = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/budgets",
            MASTER_KEY,
            Some(json!({ "max_budget": 0.01, "duration_sec": 3600, "enforce": true })),
        ))
        .await
        .unwrap();
    let budget_id = body_to_json(budget.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "user-1", "budget_id": budget_id })),
        ))
        .await
        .unwrap();

    let key_resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/keys",
            MASTER_KEY,
            Some(json!({
                "user_id": "user-1",
                "allowed_models": ["openai:*"]
            })),
        ))
        .await
        .unwrap();
    let virtual_key = body_to_json(key_resp.into_body()).await["key"]
        .as_str()
        .unwrap()
        .to_string();

    // Set spend over budget
    {
        let conn = state.db.clone();
        conn.record_usage(
            None,
            "user-1",
            "openai:gpt-4o-mini",
            1000,
            1000,
            0.05,
            "req-1",
        )
        .unwrap();
    }

    let resp = app
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "hello" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn list_models_returns_allowlist() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state.clone());

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "user-1" })),
        ))
        .await
        .unwrap();

    let key_resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/keys",
            MASTER_KEY,
            Some(json!({
                "user_id": "user-1",
                "allowed_models": ["openai:gpt-4o-mini", "anthropic:*"]
            })),
        ))
        .await
        .unwrap();
    let virtual_key = body_to_json(key_resp.into_body()).await["key"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = app
        .oneshot(auth_request("GET", "/v1/models", &virtual_key, None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    let ids: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"openai:gpt-4o-mini"));
    assert!(ids.contains(&"anthropic:*"));
}

#[tokio::test]
async fn health_endpoints_work() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state);

    let live = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(live.status(), StatusCode::OK);

    let ready = app
        .oneshot(
            Request::builder()
                .uri("/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
}
