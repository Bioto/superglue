#![cfg(feature = "capture")]

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use superglue::gateway::capture::CaptureConfig;
use superglue::gateway::{router, test_state_with_openai, test_state_with_openai_and_capture};

const MASTER_KEY: &str = "test-master-key-12345";

fn auth_request(
    method: &str,
    uri: &str,
    key: &str,
    body: Option<Value>,
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

fn capture_config(spool_dir: PathBuf) -> CaptureConfig {
    CaptureConfig {
        s3_bucket: "test-capture-bucket".into(),
        s3_prefix: "gateway-capture".into(),
        spool_dir,
        rotate_bytes: 64 * 1024 * 1024,
        rotate_secs: 3600,
        max_response_bytes: 16 * 1024 * 1024,
        exclude_users: HashSet::new(),
        aws_region: Some("us-west-2".into()),
        s3_endpoint: Some("http://127.0.0.1:9".into()),
    }
}

fn read_spool_records(spool_dir: &Path) -> Vec<Value> {
    let mut records = Vec::new();
    let Ok(entries) = fs::read_dir(spool_dir) else {
        return records;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".ndjson.gz"))
        {
            continue;
        }
        let file = fs::File::open(&path).expect("open spool");
        let mut decoder = GzDecoder::new(file);
        let mut body = String::new();
        if decoder.read_to_string(&mut body).is_err() {
            continue;
        }
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            records.push(serde_json::from_str(line).expect("parse capture ndjson"));
        }
    }
    records
}

async fn setup_user_and_key(app: &axum::Router) -> String {
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
                "allowed_models": ["openai:gpt-4o-mini", "openai:gpt-5.6-luna"]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(key_resp.status(), StatusCode::OK);
    let bytes = key_resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice::<Value>(&bytes).unwrap()["key"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn capture_disabled_when_not_configured() {
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
    assert!(state.capture.is_none());
    let app = router(state.clone());
    let virtual_key = setup_user_and_key(&app).await;
    let resp = app
        .clone()
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
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn capture_records_non_streaming_chat_completion() {
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
    let spool_dir = dir.path().join("spool");
    let state = test_state_with_openai_and_capture(
        MASTER_KEY,
        &dir.path().join("gw.db"),
        &server.uri(),
        "sk-test",
        capture_config(spool_dir.clone()),
    )
    .await;
    let app = router(state.clone());
    let virtual_key = setup_user_and_key(&app).await;
    let resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "hello capture" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    drop(app);
    drop(state);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let records = read_spool_records(&spool_dir);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["api"], "chat_completions");
    assert_eq!(records[0]["stream"], false);
    assert_eq!(records[0]["user_id"], "user-1");
    assert_eq!(records[0]["request"]["messages"][0]["content"], "hello capture");
    assert!(records[0]["response"].is_object());
}

#[tokio::test]
async fn capture_records_streaming_responses_with_raw_sse() {
    let server = MockServer::start().await;
    let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spool_dir = dir.path().join("spool");
    let state = test_state_with_openai_and_capture(
        MASTER_KEY,
        &dir.path().join("gw.db"),
        &server.uri(),
        "sk-test",
        capture_config(spool_dir.clone()),
    )
    .await;
    let app = router(state.clone());
    let virtual_key = setup_user_and_key(&app).await;
    let resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/responses",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-5.6-luna",
                "input": "stream me",
                "stream": true
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let _ = resp.into_body().collect().await.unwrap();

    drop(app);
    drop(state);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let records = read_spool_records(&spool_dir);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["api"], "responses");
    assert_eq!(records[0]["stream"], true);
    assert_eq!(records[0]["request"]["input"], "stream me");
    let sse_events = records[0]["sse"].as_array().expect("sse array");
    assert!(!sse_events.is_empty());
    assert!(sse_events.iter().any(|v| v.as_str().unwrap().contains("output_text.delta")));
}

#[tokio::test]
async fn capture_excludes_configured_users() {
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
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let spool_dir = dir.path().join("spool");
    let mut cfg = capture_config(spool_dir.clone());
    cfg.exclude_users.insert("user-1".into());
    let state = test_state_with_openai_and_capture(
        MASTER_KEY,
        &dir.path().join("gw.db"),
        &server.uri(),
        "sk-test",
        cfg,
    )
    .await;
    let app = router(state.clone());
    let virtual_key = setup_user_and_key(&app).await;
    let resp = app
        .clone()
        .oneshot(auth_request(
            "POST",
            "/v1/chat/completions",
            &virtual_key,
            Some(json!({
                "model": "openai:gpt-4o-mini",
                "messages": [{ "role": "user", "content": "skip" }]
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    drop(app);
    drop(state);
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(read_spool_records(&spool_dir).is_empty());
}
