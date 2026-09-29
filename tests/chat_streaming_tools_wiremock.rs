//! Streaming tool-round tests (OpenAI SSE format).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use async_trait::async_trait;
use serde_json::json;
use superglue::chat::{ChatOptions, stream_complete_with_tools};
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;
use superglue::http::{ClientConfig, HttpClient};
use superglue::openai::ChatMessage;
use superglue::tools::{Tool, ToolRegistry, ToolSpec};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "echo".to_string(),
            description: None,
            parameters_schema: json!({"type": "object"}),
            static_tool: false,
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, superglue::tools::ToolInvokeError> {
        Ok(json!({ "echo": arguments }))
    }
}

fn tool_round_sse() -> String {
    let mut body = String::new();
    let chunks = [
        json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": ""}
                    }]
                },
                "finish_reason": null
            }]
        }),
        json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"x\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    ];
    for c in chunks {
        body.push_str(&format!("data: {c}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn text_round_sse(tokens: &[&str]) -> String {
    let mut body = String::new();
    for t in tokens {
        let c = json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": {"content": t},
                "finish_reason": null
            }]
        });
        body.push_str(&format!("data: {c}\n\n"));
    }
    let final_c = json!({
        "id": "chatcmpl-s",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });
    body.push_str(&format!("data: {final_c}\n\n"));
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn stream_tool_round_then_text() {
    let server = MockServer::start().await;
    let n = Arc::new(AtomicU32::new(0));
    let n2 = Arc::clone(&n);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            let sse = if i == 0 {
                tool_round_sse()
            } else {
                text_round_sse(&["done"])
            };
            ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
        })
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let reg = ToolRegistry::new();
    reg.register(Arc::new(EchoTool)).await.unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 4,
        ..Default::default()
    };

    let mut deltas = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &reg,
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &opts,
        |d| deltas.push(d),
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(out.content, "done");
    assert_eq!(out.rounds, 2);
    assert_eq!(deltas.join(""), "done");
}

fn reasoning_then_text_sse() -> String {
    let mut body = String::new();
    for (field, value) in [
        ("reasoning", "first think"),
        ("reasoning_content", " second think"),
        ("content", "hello"),
    ] {
        let c = json!({
            "id": "chatcmpl-s",
            "choices": [{
                "index": 0,
                "delta": { field: value },
                "finish_reason": null
            }]
        });
        body.push_str(&format!("data: {c}\n\n"));
    }
    let final_c = json!({
        "id": "chatcmpl-s",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    });
    body.push_str(&format!("data: {final_c}\n\n"));
    body.push_str("data: [DONE]\n\n");
    body
}

#[tokio::test]
async fn stream_forwards_chat_reasoning_deltas() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(reasoning_then_text_sse(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let http = HttpClient::new(ClientConfig::default()).unwrap();
    let opts = ChatOptions {
        base_url: server.uri(),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 2,
        ..Default::default()
    };

    let mut content = Vec::new();
    let mut reasoning = Vec::new();
    let out = stream_complete_with_tools(
        &http,
        &ToolRegistry::new(),
        &HookRegistry::new(),
        &GuardrailRegistry::new(),
        vec![ChatMessage::text("user", "go")],
        &opts,
        |d| content.push(d),
        |d| reasoning.push(d),
    )
    .await
    .unwrap();

    assert_eq!(out.content, "hello");
    assert_eq!(content.join(""), "hello");
    assert_eq!(reasoning.join(""), "first think second think");
}

/// OpenRouter comment keepalives must not hold a round open with zero tokens.
#[tokio::test]
async fn provider_heartbeats_end_a_silent_tool_stream() {
    let address = spawn_sse_server(Vec::new(), true).await;
    let http = short_idle_client();
    let opts = local_opts(&address);
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        stream_complete_with_tools(
            &http,
            &ToolRegistry::new(),
            &HookRegistry::new(),
            &GuardrailRegistry::new(),
            vec![ChatMessage::text("user", "go")],
            &opts,
            |_| {},
            |_| {},
        ),
    )
    .await;
    let err = result
        .expect("stream hung on provider heartbeats")
        .unwrap_err();
    assert!(err.is_transient_stream_stall(), "{err}");
    assert!(started.elapsed() < Duration::from_secs(5));
}

/// A finished chunk plus heartbeats must complete. Do not wait for `[DONE]`.
#[tokio::test]
async fn provider_heartbeats_after_stop_finish_the_round() {
    let payload = concat!(
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n"
    );
    let address = spawn_sse_server(payload.as_bytes().to_vec(), true).await;
    let http = HttpClient::new(ClientConfig {
        stream_first_byte_timeout: Duration::from_secs(2),
        stream_idle_timeout: Duration::from_secs(5),
        retry: superglue::http::RetryPolicy {
            max_retries: 0,
            ..superglue::http::RetryPolicy::default()
        },
        ..ClientConfig::default()
    })
    .unwrap();
    let opts = local_opts(&address);
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        stream_complete_with_tools(
            &http,
            &ToolRegistry::new(),
            &HookRegistry::new(),
            &GuardrailRegistry::new(),
            vec![ChatMessage::text("user", "go")],
            &opts,
            |_| {},
            |_| {},
        ),
    )
    .await;
    let out = result
        .expect("stream hung after finish_reason")
        .expect("round should finish without [DONE]");
    assert_eq!(out.content, "hello");
}

fn local_opts(address: &std::net::SocketAddr) -> ChatOptions {
    ChatOptions {
        base_url: format!("http://{address}"),
        api_key: secrecy::SecretString::from("sk-test".to_string()),
        model: "openai:mock".into(),
        max_tool_rounds: 2,
        ..Default::default()
    }
}

fn short_idle_client() -> HttpClient {
    HttpClient::new(ClientConfig {
        stream_first_byte_timeout: Duration::from_secs(2),
        stream_idle_timeout: Duration::from_millis(80),
        retry: superglue::http::RetryPolicy {
            max_retries: 0,
            ..superglue::http::RetryPolicy::default()
        },
        ..ClientConfig::default()
    })
    .unwrap()
}

/// Write `preamble`, then `: OPENROUTER PROCESSING` until the client drops.
async fn spawn_sse_server(preamble: Vec<u8>, heartbeats: bool) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let preamble = preamble.clone();
            tokio::spawn(async move {
                let mut buf = [0_u8; 4096];
                let _ = socket.read(&mut buf).await;
                if socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
                    )
                    .await
                    .is_err()
                {
                    return;
                }
                if !preamble.is_empty() && write_chunk(&mut socket, &preamble).await.is_err() {
                    return;
                }
                if !heartbeats {
                    return;
                }
                loop {
                    if write_chunk(&mut socket, b": OPENROUTER PROCESSING\n\n")
                        .await
                        .is_err()
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(15)).await;
                }
            });
        }
    });
    address
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, data: &[u8]) -> std::io::Result<()> {
    socket
        .write_all(format!("{:x}\r\n", data.len()).as_bytes())
        .await?;
    socket.write_all(data).await?;
    socket.write_all(b"\r\n").await?;
    Ok(())
}
