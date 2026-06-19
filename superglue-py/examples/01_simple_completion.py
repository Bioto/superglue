"""Example 01: Simple text completion with no tools.

Run after `maturin develop` inside superglue-py/:

    OPENAI_API_KEY=sk-... uv run examples/01_simple_completion.py
"""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
)

result = client.complete("What is the capital of France? Reply in one sentence.")

print("content:", result.content)
print("rounds:", result.rounds)   # 1 — no tool calls involved
print("usage:", result.usage)
