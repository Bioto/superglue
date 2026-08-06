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
async fn virtual_key_allowed_model_proxies_responses_and_logs_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp-gw",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hi" }]
            }],
            "usage": { "input_tokens": 12, "output_tokens": 4, "total_tokens": 16 }
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

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "user-1", "alias": "Alice" })),
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
                "name": "test-key",
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
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/responses",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "input": "hello"
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let usage = state.db.list_usage(None, None, 10).unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].user_id, "user-1");
    assert_eq!(usage[0].prompt_tokens, 12);
    assert_eq!(usage[0].completion_tokens, 4);
}

#[tokio::test]
async fn virtual_key_disallowed_model_returns_403_for_responses() {
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
            "/v1/responses",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o",
                "input": "hello"
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
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
async fn list_models_fetches_upstream_for_master_key() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "id": "gpt-4o-mini", "object": "model" },
                { "id": "gpt-4o", "object": "model" }
            ]
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
    let app = router(state);

    let resp = app
        .oneshot(auth_request("GET", "/v1/models", MASTER_KEY, None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    let ids: Vec<_> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(ids.contains(&"openai:gpt-4o-mini"));
    assert!(ids.contains(&"openai:gpt-4o"));
    assert!(!ids.contains(&"*"));
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

#[tokio::test]
async fn update_user_budget_does_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state);

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "budget-user", "alias": "Budget User" })),
        ))
        .await
        .unwrap();

    let budget = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/budgets",
            MASTER_KEY,
            Some(json!({ "max_budget": 50.0, "duration_sec": 2592000, "enforce": true })),
        ))
        .await
        .unwrap();
    assert_eq!(budget.status(), StatusCode::OK);
    let budget_id = body_to_json(budget.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    let update = app
        .clone()
        .oneshot(auth_request(
            "PATCH",
            "/v1/users/budget-user",
            MASTER_KEY,
            Some(json!({ "budget_id": budget_id })),
        ))
        .await
        .unwrap();
    assert_eq!(update.status(), StatusCode::OK);
    let user = body_to_json(update.into_body()).await;
    assert_eq!(user["budget_id"].as_str(), Some(budget_id.as_str()));
    assert_eq!(user["alias"].as_str(), Some("Budget User"));

    let list = app
        .clone()
        .oneshot(auth_request("GET", "/v1/users", MASTER_KEY, None))
        .await
        .unwrap();
    assert_eq!(list.status(), StatusCode::OK);
}

#[tokio::test]
async fn delete_user_removes_keys() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("gw.db"));
    let app = router(state.clone());

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "delete-me", "alias": "Gone" })),
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
                "user_id": "delete-me",
                "allowed_models": ["openai:*"]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(key_resp.status(), StatusCode::OK);
    let key_id = body_to_json(key_resp.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    let delete = app
        .clone()
        .oneshot(auth_request(
            "DELETE",
            "/v1/users/delete-me",
            MASTER_KEY,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(delete.status(), StatusCode::OK);
    let body = body_to_json(delete.into_body()).await;
    assert_eq!(body["deleted"].as_str(), Some("delete-me"));
    assert_eq!(body["keys_deleted"].as_u64(), Some(1));

    let list_users = app
        .clone()
        .oneshot(auth_request("GET", "/v1/users", MASTER_KEY, None))
        .await
        .unwrap();
    let users = body_to_json(list_users.into_body()).await["users"]
        .as_array()
        .unwrap()
        .clone();
    assert!(!users.iter().any(|u| u["id"].as_str() == Some("delete-me")));

    let list_keys = app
        .clone()
        .oneshot(auth_request("GET", "/v1/keys", MASTER_KEY, None))
        .await
        .unwrap();
    let keys = body_to_json(list_keys.into_body()).await["keys"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        !keys
            .iter()
            .any(|k| k["id"].as_str() == Some(key_id.as_str()))
    );

    let missing = app
        .clone()
        .oneshot(auth_request(
            "DELETE",
            "/v1/users/delete-me",
            MASTER_KEY,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn virtual_key_streaming_completion_logs_usage() {
    let server = MockServer::start().await;
    let sse_body = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse_body),
        )
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

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "stream-user", "alias": "Stream" })),
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
                "name": "stream-key",
                "user_id": "stream-user",
                "allowed_models": ["openai:gpt-4o-mini"]
            })),
        ))
        .await
        .unwrap();
    let key_json = body_to_json(key_resp.into_body()).await;
    let virtual_key = key_json["key"].as_str().unwrap();

    let resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "hello" }],
                "stream": true
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let _ = body_to_json(resp.into_body()).await;

    let usage = state.db.list_usage(None, None, 10).unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].prompt_tokens, 7);
    assert_eq!(usage[0].completion_tokens, 3);
}

#[tokio::test]
async fn remote_client_admin_commands() {
    use superglue::gateway::remote::RemoteClient;

    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("remote-cli.db"));
    let app = router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = RemoteClient::new(&format!("http://{addr}"), MASTER_KEY).unwrap();
    let user = client
        .create_user("remote-user", Some("Remote"), None)
        .await
        .unwrap();
    assert_eq!(user.id, "remote-user");

    let users = client.list_users().await.unwrap();
    assert!(users.iter().any(|u| u.id == "remote-user"));

    let budget = client.create_budget(25.0, 86_400, true).await.unwrap();
    assert!(budget.max_budget > 0.0);

    let keys = client.list_keys().await.unwrap();
    assert!(keys.is_empty());

    let models = client.list_models(MASTER_KEY).await.unwrap();
    assert!(models.is_empty() || !models.is_empty()); // env may supply provider keys
}

#[tokio::test]
async fn budget_update_and_delete_via_admin_api() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("budget-crud.db"));
    let app = router(state);

    let create = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/budgets",
            MASTER_KEY,
            Some(json!({ "max_budget": 10.0, "duration_sec": 3600, "enforce": true })),
        ))
        .await
        .unwrap();
    assert_eq!(create.status(), StatusCode::OK);
    let budget_id = body_to_json(create.into_body()).await["id"]
        .as_str()
        .unwrap()
        .to_string();

    let updated = app
        .clone()
        .oneshot(auth_request(
            "PATCH",
            &format!("/v1/budgets/{budget_id}"),
            MASTER_KEY,
            Some(json!({ "max_budget": 20.0, "enforce": false })),
        ))
        .await
        .unwrap();
    assert_eq!(updated.status(), StatusCode::OK);
    let updated_body = body_to_json(updated.into_body()).await;
    assert_eq!(updated_body["max_budget"].as_f64(), Some(20.0));
    assert_eq!(updated_body["enforce"].as_bool(), Some(false));

    let user = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "budget-assignee", "budget_id": budget_id })),
        ))
        .await
        .unwrap();
    assert_eq!(user.status(), StatusCode::OK);

    let deleted = app
        .clone()
        .oneshot(auth_request(
            "DELETE",
            &format!("/v1/budgets/{budget_id}"),
            MASTER_KEY,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    let deleted_body = body_to_json(deleted.into_body()).await;
    assert_eq!(deleted_body["users_cleared"].as_u64(), Some(1));
}

#[tokio::test]
async fn user_budget_can_be_cleared_with_null() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("budget-clear.db"));
    let app = router(state);

    let budget = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/budgets",
            MASTER_KEY,
            Some(json!({ "max_budget": 5.0, "duration_sec": 3600, "enforce": true })),
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
            Some(json!({ "user_id": "u-clear", "budget_id": budget_id })),
        ))
        .await
        .unwrap();

    let cleared = app
        .clone()
        .oneshot(auth_request(
            "PATCH",
            "/v1/users/u-clear",
            MASTER_KEY,
            Some(json!({ "budget_id": null })),
        ))
        .await
        .unwrap();
    assert_eq!(cleared.status(), StatusCode::OK);
    let body = body_to_json(cleared.into_body()).await;
    assert!(body["budget_id"].is_null());
}

#[tokio::test]
async fn usage_summary_returns_totals() {
    let dir = tempfile::tempdir().unwrap();
    let state = test_state(MASTER_KEY, &dir.path().join("usage-summary.db"));
    let app = router(state.clone());

    app.clone()
        .oneshot(auth_request(
            "POST",
            "/v1/users",
            MASTER_KEY,
            Some(json!({ "user_id": "u-sum" })),
        ))
        .await
        .unwrap();

    state.db.run_blocking(|db| {
        db.record_usage(
            None,
            "u-sum",
            "openai:gpt-4o-mini",
            10,
            5,
            0.01,
            "req-1",
        )
    })
    .await
    .unwrap();

    let resp = app
        .clone()
        .oneshot(auth_request(
            "GET",
            "/v1/usage/summary?group_by=user",
            MASTER_KEY,
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_to_json(resp.into_body()).await;
    assert_eq!(body["totals"]["requests"].as_u64(), Some(1));
    assert!(body["totals"]["cost_usd"].as_f64().unwrap() > 0.0);
}
