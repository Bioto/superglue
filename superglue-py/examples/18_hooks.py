"""Lifecycle hooks: observe and mutate the LLM pipeline.

Hooks fire at well-defined points in every call made by the client.
This example demonstrates all seven stages and the two error strategies.

Stages
------
  HookStage.PRE_COMPLETION   – before each LLM HTTP call      (observation)
  HookStage.POST_COMPLETION  – after LLM response             (observation)
  HookStage.PRE_TOOL         – before a tool is invoked       (mutating)
  HookStage.POST_TOOL        – after a tool returns           (mutating)
  HookStage.ON_RETRY         – when an HTTP retry is queued   (observation)
  HookStage.PRE_BATCH_ITEM   – before each batch item         (observation)
  HookStage.POST_BATCH_ITEM  – after each batch item          (observation)

Run
---
    export OPENAI_API_KEY=sk-...
    python examples/18_hooks.py
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Annotated

from superglue import Client, HookStage

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    print("OPENAI_API_KEY is not set — set it and re-run.")
    raise SystemExit(1)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")


# ---------------------------------------------------------------------------
# Helper: pretty-print a hook context dict
# ---------------------------------------------------------------------------

def _fmt(ctx: dict) -> str:
    snippet = textwrap.shorten(ctx["content"], width=60, placeholder="…")
    meta = ctx.get("metadata") or {}
    parts = [f"[{ctx['stage']}] {snippet!r}"]
    if meta:
        parts.append(f"meta={meta}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Example 1 — Observation hooks (logging)
# ---------------------------------------------------------------------------

def example_observation_hooks() -> None:
    print("\n=== Example 1: Observation hooks (logging) ===")

    client = Client(api_key=API_KEY, model=MODEL)

    def on_pre_completion(ctx: dict) -> None:
        print(f"  → PRE_COMPLETION : {textwrap.shorten(ctx['content'], 60)!r}")

    def on_post_completion(ctx: dict) -> None:
        print(f"  ← POST_COMPLETION: {textwrap.shorten(ctx['content'], 60)!r}")

    client.register_hook(HookStage.PRE_COMPLETION, on_pre_completion, name="log-pre")
    client.register_hook(HookStage.POST_COMPLETION, on_post_completion, name="log-post")

    result = client.complete("In one sentence, what is the speed of light?")
    print(f"  content: {result.content!r}")


# ---------------------------------------------------------------------------
# Example 2 — Mutating PreTool hook (override args before the tool runs)
# ---------------------------------------------------------------------------

def example_pre_tool_mutation() -> None:
    print("\n=== Example 2: PreTool hook — mutate args before tool call ===")

    client = Client(api_key=API_KEY, model=MODEL)

    # Tool: returns the temperature for a city.
    def get_weather(
        city: Annotated[str, "City name"],
        unit: Annotated[str, "celsius or fahrenheit"] = "celsius",
    ) -> dict:
        """Return the current temperature for a city."""
        # Fake temperatures per city (lowercase).
        temps = {"london": 15, "paris": 18, "new york": 22, "tokyo": 25}
        temp = temps.get(city.lower(), 20)
        return {"city": city, "temperature": temp, "unit": unit}

    client.register_tool(get_weather)

    # Hook: force unit to "fahrenheit" regardless of what the model asks for.
    def force_fahrenheit(ctx: dict) -> str:
        args = json.loads(ctx["content"])
        args["unit"] = "fahrenheit"
        print(f"  [pre_tool] Overriding unit → fahrenheit  (tool: {ctx['metadata'].get('tool_name')})")
        return json.dumps(args)

    client.register_hook(HookStage.PRE_TOOL, force_fahrenheit, name="force-fahrenheit")

    result = client.complete("What is the temperature in London in celsius?")
    print(f"  content: {result.content!r}")


# ---------------------------------------------------------------------------
# Example 3 — Mutating PostTool hook (augment tool result before LLM sees it)
# ---------------------------------------------------------------------------

def example_post_tool_mutation() -> None:
    print("\n=== Example 3: PostTool hook — augment result before model sees it ===")

    client = Client(api_key=API_KEY, model=MODEL)

    def lookup_user(user_id: Annotated[str, "User ID to look up"]) -> dict:
        """Look up a user by ID."""
        return {"user_id": user_id, "name": "Alice", "role": "admin"}

    client.register_tool(lookup_user)

    # Hook: inject extra context into every tool result.
    def inject_context(ctx: dict) -> str:
        result = json.loads(ctx["content"])
        result["_fetched_at"] = "2026-04-12T00:00:00Z"
        result["_source"] = "internal-db"
        print(f"  [post_tool] Injected metadata into result")
        return json.dumps(result)

    client.register_hook(HookStage.POST_TOOL, inject_context, name="inject-meta")

    result = client.complete("Look up user 'u-42' and tell me their role.")
    print(f"  content: {result.content!r}")


# ---------------------------------------------------------------------------
# Example 4 — Abort on error
# ---------------------------------------------------------------------------

def example_abort_on_error() -> None:
    print("\n=== Example 4: Hook with error_strategy='abort' ===")

    client = Client(api_key=API_KEY, model=MODEL)

    # A hook that always fails — will abort the completion.
    def always_fail(ctx: dict) -> str:
        raise RuntimeError("Simulated hook failure")

    client.register_hook(
        HookStage.PRE_COMPLETION,
        always_fail,
        name="always-fail",
        error_strategy="abort",
    )

    try:
        client.complete("Hello!")
        print("  (should not reach here)")
    except RuntimeError as exc:
        print(f"  Caught expected error: {exc}")


# ---------------------------------------------------------------------------
# Example 5 — Skip on error (default)
# ---------------------------------------------------------------------------

def example_skip_on_error() -> None:
    print("\n=== Example 5: Hook with error_strategy='skip' (default) ===")

    client = Client(api_key=API_KEY, model=MODEL)

    def flaky_hook(ctx: dict) -> None:
        raise RuntimeError("Transient failure — will be skipped")

    # Default error_strategy is 'skip' — the call proceeds even if the hook fails.
    client.register_hook(HookStage.PRE_COMPLETION, flaky_hook, name="flaky")

    result = client.complete("What is 1 + 1?")
    print(f"  content: {result.content!r}  (call succeeded despite hook error)")


# ---------------------------------------------------------------------------
# Example 6 — Batch hooks
# ---------------------------------------------------------------------------

def example_batch_hooks() -> None:
    print("\n=== Example 6: Batch hooks (PRE_BATCH_ITEM / POST_BATCH_ITEM) ===")

    client = Client(api_key=API_KEY, model=MODEL)
    pre_count = [0]
    post_count = [0]

    def on_pre(ctx: dict) -> None:
        pre_count[0] += 1
        print(f"  [pre_batch_item #{pre_count[0]}] prompt={ctx['content'][:40]!r}")

    def on_post(ctx: dict) -> None:
        post_count[0] += 1
        print(f"  [post_batch_item #{post_count[0]}] reply={ctx['content'][:40]!r}")

    client.register_hook(HookStage.PRE_BATCH_ITEM, on_pre, name="batch-pre")
    client.register_hook(HookStage.POST_BATCH_ITEM, on_post, name="batch-post")

    resp = client.batch(
        [
            "What is the capital of France?",
            "What is the capital of Japan?",
            "What is the capital of Brazil?",
        ],
        max_concurrent=3,
    )
    print(f"  {resp.successful}/{resp.total_requests} succeeded")
    print(f"  pre fires={pre_count[0]}  post fires={post_count[0]}")


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_observation_hooks()
    example_pre_tool_mutation()
    example_post_tool_mutation()
    example_abort_on_error()
    example_skip_on_error()
    example_batch_hooks()
