# Tool boundary and async

## Contract

**Implemented**: the Rust core treats tools as abstract callbacks keyed by name with **JSON arguments** and **JSON results**. The core does not parse Python type objects or Pydantic models. Each binding is responsible for:

1. Building a **JSON Schema** that describes parameters shown to the model.
2. Dispatching execution when the core requests a tool call, including deserialization of arguments and serialization of results to JSON.
3. Reporting failures via `ToolInvokeError` so the core can propagate them uniformly.

### Rust side

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn call(&self, arguments: Value) -> Result<Value, ToolInvokeError>;
}

pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub parameters_schema: serde_json::Value,  // JSON Schema object
}
```

### Python side

```python
def get_weather(args: dict) -> dict:
    return {"temp": 22, "unit": "C"}

client.register_tool(
    name="get_weather",
    description="Get current weather for a location.",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    fn=get_weather,
)
```

The binding (`PythonDictTool`) handles `json.loads` → Python dict → `fn(args)` → `json.dumps` transparently. The Python callable receives and returns plain dicts; it never touches JSON strings.

## Async model

**Implemented**: orchestration is **async end-to-end** in the core (Tokio). The Python binding bridges via `tokio::runtime::Runtime::block_on`, blocking the calling Python thread.

**Tokio owns the event loop**: Python's asyncio is not involved. Tool callbacks are synchronous Python functions; they are dispatched via `tokio::task::spawn_blocking` which runs them on a dedicated OS thread where `Python::attach` (GIL management) is safe.

### Why `spawn_blocking` for callbacks

When `complete_with_tools` needs to call a Python tool:

```
Tokio async loop (Python main thread, blocked in block_on)
  └── spawn_blocking → OS thread
        └── Python::attach → calls fn(args: dict) → returns dict
              └── sends result back to async loop via JoinHandle
```

This avoids calling Python from within an async polling context, which would crash on free-threaded CPython (no thread state for the polling context).

### Streaming token delivery

The same constraint applies to streaming token callbacks. See [Architecture](02-architecture.md#token-delivery-in-streaming) for the `mpsc` channel pattern used.

## Tool registry

`ToolRegistry` is an `Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>` protected for concurrent access. Tools are registered once and dispatched many times concurrently. The registry is cloned (cheap `Arc` clone) into each `complete_with_tools` call.

## Testing strategy

**Implemented**: the `tools::harness` module drives the tool loop with in-memory JSON callbacks. Integration tests in `tests/chat_tool_loop_wiremock.rs` use `wiremock` to serve fake LLM responses that include tool calls, exercising the full round-trip without a live model.

```rust
// Offline harness test (no network)
let plan = vec![
    ToolInvocation { name: "echo".into(), arguments: json!({"x": 1}) },
];
let results = run_plan(&registry, plan).await;
```

## Error handling

`ToolInvokeError` has three variants:

- `NotFound(name)` — no tool registered with that name.
- `Handler(msg, source)` — the tool function returned an error.
- `Serialization(msg)` — JSON encode/decode failed.

The `complete_with_tools` loop propagates `ToolInvokeError` as `ChatError::Tool`, aborting the run. Per-tool retry policy is an **open decision** for a future release.

## See also

- [Architecture](02-architecture.md) for the split between core and bindings.
- [Protobuf, gRPC, and workflows](04-protobuf-grpc-and-workflows.md) for how tool specs appear in the proto schema.
