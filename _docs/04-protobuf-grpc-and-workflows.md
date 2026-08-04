# Protobuf, gRPC, and workflows

## Current status

**Protobuf schema is implemented**. **gRPC server is implemented** (feature-gated by `--features grpc`). **Audit resume/replay helpers are shipped** (`RunStore`, `RunRecorder`, `resume_chat`, `resume_response`). Workflow engine remains planned.

## Why protobuf internally

Workflows, hooks, configuration, and internal events are richer than ad hoc JSON blobs. A **canonical binary schema** provides:

- Stable, versioned contracts between components and across language bindings.
- Efficient serialization for high-volume event streams.
- Generated types in multiple languages from the same `.proto` definitions.

**Decided direction**: protobuf is the internal source of truth for `ChatOptions`, `ChatMessage`, `Usage`, `ToolSpec`, `CompletionOutcome`, and related types. `prost` generates Rust; future language clients can generate from the same `.proto`.

JSON remains appropriate for **tool arguments and results** at the host boundary (see [Tool boundary and async](03-tool-boundary-and-async.md)).

## Implemented schema (`proto/superglue.proto`)

```protobuf
message ChatOptions {
  string base_url         = 1;
  string api_key          = 2;
  string model            = 3;
  uint32 max_tool_rounds  = 4;
  optional string system_prompt = 5;

  // Sampling
  optional float  temperature         = 6;
  optional float  top_p               = 7;
  optional uint32 max_completion_tokens = 8;

  // Penalties
  optional float  presence_penalty    = 9;
  optional float  frequency_penalty   = 10;

  // Tool control
  optional bool   parallel_tool_calls = 11;

  // Logprobs
  optional bool   logprobs            = 12;
  optional uint32 top_logprobs        = 13;

  // Determinism / storage
  optional int64  seed                = 14;
  optional bool   store               = 15;

  // Service / reasoning
  optional string service_tier        = 16;
  optional string reasoning_effort    = 17;
  optional string stop                = 18;
  optional string extra_json          = 19;  // generic passthrough
}

message ChatMessage {
  string role             = 1;
  optional string content = 2;
  optional string name    = 3;
  optional string tool_call_id = 4;
  optional string refusal = 6;
}

message Usage {
  uint32 prompt_tokens     = 1;
  uint32 completion_tokens = 2;
  uint32 total_tokens      = 3;
}

message ToolSpec {
  string name              = 1;
  string parameters_schema = 2;  // JSON Schema as string
  optional string description = 3;
}

message CompletionOutcome {
  optional string content       = 1;
  uint32          rounds        = 2;
  optional Usage  usage         = 3;
  optional string finish_reason = 4;
}
```

## How proto types are used today

The generated `proto::` types serve two roles:

1. **Data representation**: `CompletionOutcome` carries `usage: Option<proto::Usage>` through the chat pipeline and out to language bindings.
2. **Configuration bridge**: `From<proto::ChatOptions> for ChatOptions` allows configuration to be deserialized from protobuf and handed to the Rust chat functions — the path a gRPC server would use.

## gRPC — implemented (`grpc` feature)

The `tonic`-based gRPC server lives in `src/grpc/mod.rs` and is compiled with `--features grpc`. The deployment topology:

```
Host language  →  gRPC client (generated)  →  tonic server  →  Rust core
```

Two RPCs are exposed:

| RPC | Type | Delegates to |
|-----|------|-------------|
| `Complete` | Unary | `chat::complete_with_tools` |
| `Stream` | Server-streaming | `chat::stream_complete` |
| `CompleteResponses` | Unary | `responses::complete_with_tools` |
| `StreamResponses` | Server-streaming | `responses::stream_complete_with_tools` |

Start the server with `superglue serve --addr 0.0.0.0:50051 --api-key $KEY`.

Proto-generated Python/JS client stubs: run [`scripts/generate-proto-clients.sh`](../scripts/generate-proto-clients.sh) (Python via `grpcio-tools`; JS via `superglue-js/generated/client.ts` + `@grpc/proto-loader`). Rust tonic client is built with `--features grpc`.

## Deployment modes

| Mode | Status | Notes |
|------|--------|-------|
| **In-process FFI (PyO3)** | ✅ shipped | `superglue-py`; targets CPython 3.14t free-threaded |
| **gRPC / tonic sidecar** | ✅ shipped | `--features grpc`; `Complete`, `Stream`, `CompleteResponses`, `StreamResponses` |
| **Embedded library** | ✅ shipped | `superglue` crate consumed directly by Rust callers |

## Workflows and hooks

Hook and workflow transitions will be representable in the protobuf model so that:

- A run can be **serialized** for resume or audit.
- **Replay** and debugging can operate on recorded protobuf streams.

**Shipped today:** in-memory audit (`RunStore` / `RunRecorder`), `ClientBuilder::audit`, `resume_chat`, `resume_response`, and `export_record_json` / `import_record_json` for metadata-only replay files. Full workflow orchestration remains planned.

Persistence policy (in-memory only vs. disk vs. external store) will be a configuration matter, consistent with the security tier decisions in [Security and threat model](06-security-and-threat-model.md).

## See also

- [Observability](05-observability.md) for exporting traces that span gRPC and tool calls.
- [Roadmap and phasing](09-roadmap-and-phasing.md) for when to land gRPC relative to current state.
