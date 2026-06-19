"""Example 22: Process events via StatusEmitter (gluellm #351 parity).

Typed observability events during LLM calls:

- ``llm_call_start`` / ``llm_call_end`` / ``llm_call_error``
- ``tool_call_start`` / ``tool_call_end`` (when tools are registered)

``llm_call_end`` includes ``usage``, ``tool_call_count``, and
``estimated_cost_usd`` (see also ``examples/05_usage_tracking.py``).

Use on:
- ``Client(status_emitter=...)`` / ``GlueLLM(status_emitter=...)``
- ``SimpleExecutor(on_status=..., sinks=[...])`` (see ``examples/21_executors.py``)

    OPENAI_API_KEY=sk-... uv run examples/22_status_events.py
"""

import os

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

emitter = superglue.StatusEmitter()


def on_status(event: superglue.ProcessEvent) -> None:
    cost = event.estimated_cost_usd
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    extra = ""
    if event.error_type:
        extra = f" error={event.error_type!r}"
    elif event.tool_call_count:
        extra = f" tools={event.tool_call_count}"
    print(f"[{event.kind}] model={event.model} round={event.round} cost={cost_str}{extra}")


emitter.subscribe(on_status)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    status_emitter=emitter,
)

result = client.complete("Say hello in one word.")
print("Response:", result.content)
