//! gRPC service implementation for the superglue core.
//!
//! This module is compiled only when the `grpc` feature is enabled. It
//! implements the `SuperglueService` defined in `proto/superglue.proto` using
//! `tonic`.
//!
//! # Starting the server
//!
//! ```rust,no_run
//! # #[cfg(feature = "grpc")]
//! # async fn run() {
//! use std::sync::Arc;
//! use superglue::grpc::serve;
//! use superglue::guardrails::GuardrailRegistry;
//! use superglue::hooks::HookRegistry;
//! use superglue::http::{ClientConfig, HttpClient};
//! use superglue::tools::ToolRegistry;
//!
//! let http = Arc::new(HttpClient::new(ClientConfig::default()).unwrap());
//! let tools = Arc::new(ToolRegistry::new());
//! let hooks = Arc::new(HookRegistry::new());
//! let guardrails = Arc::new(GuardrailRegistry::new());
//!
//! serve("0.0.0.0:50051", http, tools, hooks, guardrails).await.unwrap();
//! # }
//! ```

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{error, info};

use crate::chat::{ChatOptions, complete_with_tools, stream_complete};
use crate::guardrails::GuardrailRegistry;
use crate::hooks::HookRegistry;
use crate::http::HttpClient;
use crate::openai::ChatMessage;
use crate::proto;
use crate::proto::superglue_service_server::{SuperglueService, SuperglueServiceServer};
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// Service struct
// ---------------------------------------------------------------------------

/// Holds the shared core resources for all RPC calls.
pub struct SuperglueGrpcService {
    http: Arc<HttpClient>,
    tools: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
}

impl SuperglueGrpcService {
    pub fn new(
        http: Arc<HttpClient>,
        tools: Arc<ToolRegistry>,
        hooks: Arc<HookRegistry>,
        guardrails: Arc<GuardrailRegistry>,
    ) -> Self {
        SuperglueGrpcService {
            http,
            tools,
            hooks,
            guardrails,
        }
    }
}

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl SuperglueService for SuperglueGrpcService {
    /// Non-streaming completion with optional tool loop.
    async fn complete(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<proto::CompletionOutcome>, Status> {
        let req = request.into_inner();
        let opts = req
            .options
            .map(ChatOptions::from)
            .unwrap_or_else(|| ChatOptions::default());

        let request_id = opts.request_id.clone().unwrap_or_else(|| "-".to_string());
        info!(%request_id, rpc = "Complete", "gRPC completion request");

        let messages: Vec<ChatMessage> = req
            .messages
            .into_iter()
            .map(chat_message_from_proto)
            .collect();

        let outcome = complete_with_tools(
            &self.http,
            &self.tools,
            &self.hooks,
            &self.guardrails,
            messages,
            &opts,
        )
        .await
        .map_err(|e| {
            error!(error = %e, %request_id, rpc = "Complete", "complete_with_tools failed");
            Status::internal(e.to_string())
        })?;

        Ok(Response::new(proto::CompletionOutcome {
            content: outcome.content,
            rounds: outcome.rounds,
            usage: outcome.usage,
            finish_reason: outcome.finish_reason,
        }))
    }

    /// Server-streaming completion.
    type StreamStream =
        Pin<Box<dyn futures_util::Stream<Item = Result<proto::StreamChunk, Status>> + Send>>;

    async fn stream(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let req = request.into_inner();
        let opts = req
            .options
            .map(ChatOptions::from)
            .unwrap_or_else(|| ChatOptions::default());

        let request_id = opts.request_id.clone().unwrap_or_else(|| "-".to_string());
        info!(%request_id, rpc = "Stream", "gRPC streaming completion request");

        let messages: Vec<ChatMessage> = req
            .messages
            .into_iter()
            .map(chat_message_from_proto)
            .collect();

        let (tx, rx) = mpsc::channel::<Result<proto::StreamChunk, Status>>(64);

        let http = Arc::clone(&self.http);
        let hooks = Arc::clone(&self.hooks);
        let guardrails = Arc::clone(&self.guardrails);

        tokio::spawn(async move {
            let tx_delta = tx.clone();
            let on_delta = move |delta: String| {
                let _ = tx_delta.try_send(Ok(proto::StreamChunk {
                    delta,
                    usage: None,
                    finish_reason: None,
                }));
            };

            match stream_complete(&http, &hooks, &guardrails, messages, &opts, on_delta).await {
                Ok(outcome) => {
                    // Send a final chunk with usage and finish_reason.
                    let _ = tx
                        .send(Ok(proto::StreamChunk {
                            delta: String::new(),
                            usage: outcome.usage,
                            finish_reason: outcome.finish_reason,
                        }))
                        .await;
                }
                Err(e) => {
                    error!(
                        error = %e,
                        %request_id,
                        rpc = "Stream",
                        "stream_complete failed"
                    );
                    let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }
}

// ---------------------------------------------------------------------------
// Proto → openai type conversion helpers
// ---------------------------------------------------------------------------

fn chat_message_from_proto(p: proto::ChatMessage) -> ChatMessage {
    use crate::openai::MessageContent;
    ChatMessage {
        role: p.role,
        content: p.content.map(MessageContent::Text),
        tool_calls: None,
        tool_call_id: p.tool_call_id,
        name: p.name,
        refusal: p.refusal,
    }
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the tonic gRPC server on `addr`.
///
/// This function blocks until the server shuts down.
///
/// # Errors
///
/// Returns a `tonic::transport::Error` if the server fails to bind or run.
pub async fn serve(
    addr: &str,
    http: Arc<HttpClient>,
    tools: Arc<ToolRegistry>,
    hooks: Arc<HookRegistry>,
    guardrails: Arc<GuardrailRegistry>,
) -> Result<(), tonic::transport::Error> {
    let addr = addr.parse().expect("invalid gRPC listen address");
    let svc = SuperglueGrpcService::new(http, tools, hooks, guardrails);

    tracing::info!(%addr, "superglue gRPC server starting");

    tonic::transport::Server::builder()
        .add_service(SuperglueServiceServer::new(svc))
        .serve(addr)
        .await
}
