# Context optimization (GlueLLM parity)

Superglue ports [GlueLLM](https://github.com/Bioto/glue-llm) context optimization as **opt-in** features on [`ChatOptions`](../src/chat/mod.rs). Defaults match GlueLLM: standard tool mode, condensing off, AAAK off.

## Recommended settings for multi-tool agents

For agents that register many tools or run multi-round tool loops, enable context optimization even if you keep GlueLLM-compatible defaults elsewhere:

| Setting | Why |
|---------|-----|
| `tool_mode = Dynamic` | Sends only the router + static tools on early rounds; matched schemas afterward. Cuts prompt size and improves time-to-first-token when you have a large tool registry. |
| `condense_tool_messages = true` | Replaces each assistant + N tool messages with one compact user summary after every tool batch. Shrinks history on later LLM rounds. |

**Trade-off:** dynamic routing adds one small routing LLM call up front. Net win when tool count and conversation depth are large; skip for single-tool or one-shot workloads.

Pin always-needed tools with `static_tool = true` so they stay available in dynamic mode without routing.

## Features

| Feature | Option | What it does |
|---------|--------|--------------|
| **Dynamic tool routing** | `tool_mode = Dynamic` | Exposes router tool `request_tools` first; a fast model selects relevant tools; only matched schemas are sent on later rounds |
| **Static tools** | `ToolSpec.static_tool = true` | Always included in dynamic mode (GlueLLM `@static_tool`) |
| **Tool-round condensing** | `condense_tool_messages = true` | After each tool round, replaces assistant + N tool messages with one compact user summary |
| **AAAK tool encoding** | `aaak_tool_condensing = true` | Uses deterministic `[AT]` blocks instead of plain `[Tool Results]` (requires condensing) |
| **History compression** | `summarize_context.enabled = true` | LLM-compresses older messages when `messages.len() > threshold`, keeping the tail |

## Dynamic routing flow

1. Register tools on the client as usual.
2. Set `tool_mode` to `dynamic` (Python: `tool_mode="dynamic"`, JS: `toolMode: "dynamic"`).
3. First LLM round sees `request_tools` plus any `static_tool` tools.
4. When the model calls `request_tools`, superglue runs `resolve_tool_route` (secondary LLM call) and injects matched tool schemas — the router is **not** executed as a real tool.
5. On routing failure, all dynamic tools are included (never bricks the agent).

Emit `ProcessEventKind::ToolRoute` with metadata `route_query` and `matched_tools` when a `StatusEmitter` is attached.

## Configuration reference

### Rust (`ClientBuilder` / `ChatOptions`)

```rust
use superglue::context::SummarizeContextConfig;
use superglue::tools::ToolMode;

let client = superglue::Client::builder()
    .tool_mode(ToolMode::Dynamic)
    .tool_route_model("gpt-4o-mini") // optional; default fast model in core
    .condense_tool_messages(true)
    .aaak_tool_condensing(false)
    .summarize_context(SummarizeContextConfig {
        enabled: true,
        threshold: 20,
        keep_recent: 6,
    })
    .aaak_compression_enabled(true)
    .aaak_compression_model(None)
    .build()?;
```

Register a pinned tool with `ToolSpec::new(...).with_static_tool(true)` or `static_tool: true` on the struct.

### Python

```python
client = superglue.Client(
    api_key="...",
    tool_mode="dynamic",
    condense_tool_messages=True,
    aaak_tool_condensing=False,
    summarize_context_enabled=True,
    summarize_context_threshold=20,
    summarize_context_keep_recent=6,
)
client.register_tool(get_time, static_tool=True)
```

### JavaScript

```javascript
const client = createClient({
  apiKey: "...",
  toolMode: "dynamic",
  condenseToolMessages: true,
});
await registerTool(client, { name: "get_time", ..., staticTool: true });
```

### Kotlin

```kotlin
val cfg = ClientConfig(
    apiKey = "...",
    toolMode = "dynamic",
    condenseToolMessages = true,
)
client.registerTool("get_time", "UTC time", "{}", callback, staticTool = true)
```

## Examples

- Rust: `cargo run --example 32_context_optimization`
- Python: `uv run examples/32_context_optimization.py`
- JS: `node examples/32_context_optimization.mjs`

Integration tests: `tests/context_optimization_wiremock.rs`.

## Benchmarks

Deterministic wiremock comparison (no API key). Uses **GlueLLM-style sequential tool chains** over **9 tools** (weather, forecast, flights, hotel, calculate, FX, translate, country + pinned `get_time`):

```bash
cargo run --example 33_context_optimization_benchmark
```

Live provider comparison (requires `OPENAI_API_KEY` + valid `OPENAI_MODEL`). Runs **short_chain** (3-tool parallel batch) and **long_chain** (6-tool parallel batch):

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # or any model your key supports
cargo run --example 34_context_optimization_live_benchmark
```

Encoding microbenches (Criterion):

```bash
cargo bench --bench context_optimization
```

Regression ordering checks:

```bash
cargo test --test context_optimization_benchmark
```

### Benchmark columns

Both examples **33** (wiremock) and **34** (live) print the same column semantics:

| Column | Meaning |
|--------|---------|
| `llm_rounds` | Number of LLM API calls in the tool loop |
| `tool_calls` | Non-router tool calls observed in final messages |
| `peak_msgs` | Max messages visible to any single LLM request (wiremock measured; live uses final count) |
| `peak_tools` | Max tools array length on any single LLM request |
| `cum_total` | **Headline metric** — sum of `total_tokens` across all LLM rounds (live) or request-size estimate (wiremock) |
| `cum_prompt` | Sum of prompt tokens across all LLM rounds |
| `vs_standard` | Delta vs `*_standard` (`-N%` = fewer tokens, `+N%` = more) |
| `done` | Whether all expected tools for the scenario were invoked (`Y`/`N`) |

**Why cumulative tokens, not final history bytes:** GlueLLM's benchmark compares total prompt+completion tokens across a multi-step tool chain. Single-round tasks and final-history snapshots do not show the compounding win from condensing + dynamic routing. Our workload uses parallel tool batches over 9 registered tools (models may still choose sequential calls live; wiremock scripts one parallel batch per scenario).

**Trust boundaries:**

- **Example 33 (wiremock)** — deterministic; use for ordering invariants and relative ranking. Token counts approximate `request_body_len / 4 + 50` per call.
- **Example 34 (live)** — informational; model behavior varies. `peak_tools` on live is expected (9 for standard, matched+static for dynamic), not measured from requests.

Rows with excess tool calls vs the expected chain show `*` on `vs_standard` with a footnote.

### Multi-turn context growth

Examples **33** and **34** also print a **multi-turn conversation** section after the single-turn tables. This is the benchmark that demonstrates bounded context for long conversations.

- **12 user turns** with history fed forward (`outcome.messages` + next prompt).
- **Standard/dynamic** configs: no summarize (history grows with each turn).
- **Condense configs**: `summarize_context` enabled (`threshold: 12`, `keep_recent: 4`) plus per-round condensing.
- Tool turns at indices 2, 5, 8, 10 inject weather tool calls; wiremock returns deterministic summarize/AAAK responses.

| Column | Meaning |
|--------|---------|
| `final_msgs` | Message count in history after all turns |
| `final_tokens` | Approx serialized history size / 4 at end |
| `peak_tokens` | Max approx context tokens at start of any turn |
| `cum_total` | Sum of all LLM request tokens across turns |
| `vs_standard` | Delta vs `multiturn_standard` (`-N%` = smaller final context, `+N%` = larger) |
| `ctx_sizes` | Per-turn message count at start of each turn (progression) |

**When to trust which table:**

- **Single-turn** — cumulative tokens across one tool-loop completion; shows per-turn overhead (dynamic routing adds rounds).
- **Multi-turn** — final/peak context size; shows condense + summarize keeping history bounded vs standard growth.

## Reference

- [GlueLLM README — Context condensing + Dynamic routing](https://github.com/Bioto/glue-llm/blob/main/README.md)
- [AAAK_COMPRESSION.md](https://github.com/Bioto/glue-llm/blob/main/docs/AAAK_COMPRESSION.md)
