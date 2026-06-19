"""Example 05: Token usage and cost tracking via StatusEmitter.

Two ways to track cost:

1. **Raw usage** — ``result.usage`` from the provider (prompt / completion tokens).
2. **Estimated USD** — ``ProcessEvent.estimated_cost_usd`` on ``llm_call_end`` events
   emitted by the built-in costing module (no manual rate tables needed).

    OPENAI_API_KEY=sk-... uv run examples/05_usage_tracking.py
"""

import os

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

emitter = superglue.StatusEmitter()
metrics: dict[str, float | int] = {"cost_usd": 0.0, "calls": 0}


def track_cost(event: superglue.ProcessEvent) -> None:
    if event.kind != "llm_call_end":
        return
    metrics["calls"] = int(metrics["calls"]) + 1
    if event.estimated_cost_usd is not None:
        metrics["cost_usd"] = float(metrics["cost_usd"]) + event.estimated_cost_usd
    usage = event.usage or {}
    print(
        f"  llm_call #{metrics['calls']}: "
        f"tokens={usage.get('total_tokens', '?')} "
        f"est=${event.estimated_cost_usd or 0:.6f}"
    )


emitter.subscribe(track_cost)

client = superglue.Client(
    api_key=api_key,
    model=model,
    status_emitter=emitter,
)

prompts = [
    "Name three programming languages invented before 1980.",
    "What is the difference between TCP and UDP? One sentence.",
    "Give me a haiku about Rust.",
]

print("Completions (live cost events):\n")
total_prompt = 0
total_completion = 0

for prompt in prompts:
    print(f"Q: {prompt}")
    result = client.complete(prompt)
    usage = result.usage or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    total_prompt += pt
    total_completion += ct
    print(f"A: {result.content}\n")

print(
    f"Token totals: {total_prompt} prompt + {total_completion} completion "
    f"= {total_prompt + total_completion}"
)
print(f"Session estimated cost (from StatusEmitter): ${metrics['cost_usd']:.6f}")
