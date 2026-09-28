# Examples (Kotlin)

Mirror [`../superglue-py/examples/`](../../superglue-py/examples/) and [`../superglue-js/examples/`](../../superglue-js/examples/) by number.

## Runnable JUnit examples (22–24)

Live sections require `OPENAI_API_KEY` and a valid `OPENAI_MODEL` (set both in the shell or rely on repo `.env`):

```bash
set -a && source ../../.env && set +a   # from superglue-kt/
./gradlew :superglue-android:testDebugUnitTest --tests 'com.superglue.kt.examples.FeatureExamplesTest'
```

Gradle unit tests also read the repo root `.env` for any unset `OPENAI_*` variables (same as `run-all-superglue-examples.sh`).

| # | Topic | Test |
|---|--------|------|
| 22 | StatusEmitter / process events | `example22_status_events` |
| 23 | `reasoningEffort` | `example23_reasoning_effort` |
| 24 | Connection pool defaults | `example24_connection_pool_defaults` |
| 25 | Model fallback | `example25_model_fallback` |
| 26 | Stream response | `example26_stream_response` |
| 27 | Complete response | `example27_complete_response` |
| 29 | Multi-provider `provider:model` | `example29_multi_provider` |
| 30 | Streaming with tool rounds | `example30_streaming_tools` |
| 31 | Inline file message | `example31_file_upload_inline` |

## API surface

- `ClientConfig.connectTimeoutSecs` default **30**, `poolMaxIdlePerHost` default **50**
- `ClientConfig.reasoningEffort` — client-level reasoning effort
- `ClientConfig.apiKeys` / `requestsPerSecondFor` / `maxUploadBytes` — multi-provider credentials and limits
- `AgentRunSpec.reasoningEffort` — per-agent override
- `SuperglueClient.open(config, statusEmitter = …)` — attach a `StatusEmitter`
- `client.complete(..., reasoningEffort = "high")` — per-call override
- `client.uploadFile` / `messageWithFileBytes` — file attachments

Full numbered examples (01–21) follow the same patterns as the JS/Python sets; run them from an Android instrumented test or internal debug screen with `SuperglueClient` / `ClientConfig`.
