"""Example 23: Per-agent reasoning effort (gluellm #361 parity).

``reasoning_effort`` controls how much internal reasoning OpenAI o-series and
gpt-5.x models apply. Values: ``"none"`` | ``"minimal"`` | ``"low"`` | ``"medium"``
| ``"high"`` | ``"xhigh"``.

Superglue normalizes effort per model family (e.g. o-series downgrades unsupported
levels to the nearest lower supported value) before sending the API request.

Set on:
- :class:`Client` / :class:`Agent` constructor
- :class:`AgentSpec.reasoning_effort` (agent override)
- Per-call: ``client.complete(..., reasoning_effort="high")``
- :class:`SimpleExecutor(reasoning_effort=...)``

gluellm alias: ``client.response(...)`` is the same as ``client.complete(...)``.

    OPENAI_API_KEY=sk-... uv run examples/23_reasoning_effort.py
"""

import os

from superglue import Agent, AgentSpec, Client, SimpleExecutor

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

# Use a reasoning-capable model (gpt-5.x or o-series). Override via env if needed.
model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

print("=" * 60)
print("1. Client-level reasoning_effort")
print("=" * 60)

client = Client(
    api_key=api_key,
    model=model,
    reasoning_effort="medium",
    system_prompt="Answer in one short sentence.",
)
# gluellm-compatible alias
result = client.response("What is 17 + 25?")
print("response():", result.content)
print()

print("=" * 60)
print("2. Per-call override")
print("=" * 60)

result = client.complete(
    "Name one benefit of higher reasoning effort.",
    reasoning_effort="high",
)
print("complete(high):", result.content)
print()

print("=" * 60)
print("3. AgentSpec.reasoning_effort")
print("=" * 60)

spec = AgentSpec(
    name="DeepThinker",
    persona="a careful analyst who thinks step-by-step",
    goals=["Give precise, well-reasoned answers"],
    reasoning_effort="high",
)

agent = Agent(spec, api_key=api_key, model=model)
result = agent.run("Why might exponential backoff beat fixed-interval retry?")
print("Agent (spec reasoning_effort=high):", result.content)
print()

print("=" * 60)
print("4. SimpleExecutor forwarding")
print("=" * 60)

ex = SimpleExecutor(
    api_key=api_key,
    model=model,
    reasoning_effort="low",
    system_prompt="Be brief.",
)
result = ex.execute("What is the speed of light in vacuum?")
print("Executor (low effort):", result.content)
print()

print("All reasoning effort examples completed.")
