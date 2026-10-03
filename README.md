# SuperGlue

**One Rust core for LLM orchestration.** Python, JavaScript, and Kotlin call the same loop.

![CI](https://github.com/Bioto/superglue/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-BSD--4--Clause-blue.svg)

SuperGlue talks to model providers, runs the tool loop, streams tokens, and records usage.
Bindings stay thin. Retry rules, SSE parsing, and tool dispatch live in one place.

```rust
let client = superglue::Client::builder()
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model("openai:gpt-4o-mini")
    .build()?;

let outcome = client
    .complete("What is the capital of France?", superglue::CallOptions::default())
    .await?;

println!("{}", outcome.content.unwrap_or_default());
```



## Why SuperGlue

Most language SDKs reimplement the same glue: HTTP, retries, tool rounds, streaming, and redaction.

That copy drifts. A retry bug in Python does not match the bug in Node.

SuperGlue puts that loop in Rust. Host languages register tools and callbacks.
Tool arguments and results cross the boundary as JSON.

## Features

- **Completions and streaming** — buffered `complete` and SSE `stream` on the same client
- **Tool loop** — JSON tools, parallel dispatch in a round, per-tool FailFast / Skip / Retry
- **Providers** — `provider:model` routing (OpenAI, Anthropic, xAI, Groq, OpenRouter, and more)
- **Reliability** — retries, rate limits, model fallback, cooperative cancel
- **Agents** — personas, hooks, guardrails, status events, usage and cost
- **Context control** — tool routing, round condensing, optional code-mode tools (`code` feature)
- **Observability** — `tracing` with redaction; optional OTLP and Prometheus
- **Optional servers** — gRPC sidecar (`grpc`) and an OpenAI-compatible HTTP gateway (`gateway`)
- **Language bindings** — Python (PyO3), Node.js (napi-rs), Kotlin (JNI)

The workflow engine is not shipped yet. See `[_docs/09-roadmap-and-phasing.md](_docs/09-roadmap-and-phasing.md)`.

## Install



### Rust

Add the crate from git (current version is `0.1.0`):

```toml
[dependencies]
superglue = { git = "https://github.com/Bioto/superglue" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Optional features stay off by default. Enable them when you need the extra surface:


| Feature               | What it adds                                       |
| --------------------- | -------------------------------------------------- |
| `mcp`                 | MCP tool clients                                   |
| `grpc`                | gRPC server and client                             |
| `gateway`             | HTTP LLM gateway (keys, budgets, usage)            |
| `otlp` / `prometheus` | OTLP spans and a metrics endpoint                  |
| `code`                | Programmatic JavaScript tool mode                  |
| `realtime`            | Realtime voice sessions (xAI WebSocket first)      |
| `capture`             | Gateway traffic capture (`gateway` plus S3 upload) |




### Python

Build the wheel from `superglue-py/` with [maturin](https://www.maturin.rs/):

```bash
cd superglue-py
./dev.sh
```

Then:

```python
import os
import superglue

client = superglue.Client(api_key=os.environ["OPENAI_API_KEY"], model="openai:gpt-4o-mini")
result = client.complete("What is the capital of France? Reply in one sentence.")
print(result.content)
```



### JavaScript and Kotlin


| Binding | Path                            | Notes                                             |
| ------- | ------------------------------- | ------------------------------------------------- |
| Node.js | `[superglue-js/](superglue-js)` | napi-rs addon; tools need an explicit JSON Schema |
| Kotlin  | `[superglue-kt/](superglue-kt)` | JNI; same core, same version                      |


All packages share `[workspace.package].version` in `[Cargo.toml](Cargo.toml)`. See [PACKAGES.md](PACKAGES.md).

## Quick start (from this repo)

1. Set a provider key. Default examples use `OPENAI_API_KEY`.
2. From the SuperGlue root, run:

```bash
cargo build
cargo test
cargo run --example 01_simple_completion
```

MCP tests need the `mcp` feature:

```bash
cargo test --features mcp
```

Numbered examples cover tools, streaming, batch, agents, MCP, and more.
Start at `[examples/README.md](examples/README.md)`.

## Providers and secrets

Pass `provider:model` (for example `openai:gpt-4o-mini` or `anthropic:claude-sonnet-4-20250514`).

Keep keys in the environment. Do not put them in config files, logs, or commits.
The core wraps keys in `secrecy` types and redacts them in traces by default.


| Provider   | API key env          |
| ---------- | -------------------- |
| OpenAI     | `OPENAI_API_KEY`     |
| Anthropic  | `ANTHROPIC_API_KEY`  |
| xAI        | `XAI_API_KEY`        |
| Groq       | `GROQ_API_KEY`       |
| OpenRouter | `OPENROUTER_API_KEY` |


Optional base URL overrides and file-upload notes live in `[docs/multi-provider.md](docs/multi-provider.md)`.

## Gateway

The `gateway` feature starts an OpenAI-compatible HTTP proxy with virtual keys, budgets, and usage tracking.

```bash
cargo build --features gateway --release

export GATEWAY_MASTER_KEY="replace-me"
export OPENAI_API_KEY="sk-..."

superglue gateway serve --addr 127.0.0.1:8080 --db ./superglue-gateway.db
```

Full CLI and threat notes: `[_docs/12-llm-gateway.md](_docs/12-llm-gateway.md)`.

## Architecture

```
Python / JS / Kotlin / CLI
            │  JSON tools, native callbacks
            ▼
     SuperGlue (Rust)
            │  HTTP + SSE
            ▼
     Model providers
```

- Orchestration runs in Rust (Tokio).
- Bindings do not reimplement the tool loop.
- Protobuf (`proto/superglue.proto`) is the shared schema for gRPC and audit records.

Maintainer design notes: `[_docs/README.md](_docs/README.md)`.

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

After you change `[workspace.package].version`:

```bash
./scripts/sync-superglue-versions.sh
./scripts/check-superglue-version-parity.sh
```

In [thestack](https://github.com/Bioto/thestack), this repo is the `projects/superglue` submodule. Layout notes: [MONOREPO.md](MONOREPO.md).

## License

BSD-4-Clause. See [LICENSE](LICENSE).

Public materials that mention this software must display the acknowledgement in the LICENSE file.
This license is GPL-incompatible.