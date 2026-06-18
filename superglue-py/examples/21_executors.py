"""Example 21 — Executor pattern.

Executors are composable, hook-wrapped execution units that mirror
``gluellm``'s executor concept.  They expose a single ``.execute(query)``
method and can be dropped into workflows, pipelines, or used standalone.

Three concrete types are available:

* ``SimpleExecutor`` — wraps a ``Client`` directly (no AgentSpec).
* ``AgentExecutor`` — wraps an ``Agent`` (persona, goals, constraints, tools).
* ``AgentStructuredExecutor`` — like ``AgentExecutor`` but parses the model's
  JSON reply into a typed dataclass or Pydantic model.

Executor-level hooks fire at the **executor boundary** (before the query
reaches the LLM pipeline and after the final text comes back), independently
of the in-pipeline hooks registered on the underlying Client or Agent.

**Status callbacks** (``on_status`` / ``sinks``) receive :class:`ProcessEvent`
objects during LLM calls — see also ``examples/22_status_events.py``.

Prerequisites
-------------
Set ``OPENAI_API_KEY`` in your environment.

Run
---
    python examples/21_executors.py
"""

from __future__ import annotations

import dataclasses
import os

from superglue import (
    Agent,
    AgentExecutor,
    AgentSpec,
    AgentStructuredExecutor,
    ProcessEvent,
    SimpleExecutor,
    StructuredOutcome,
)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY before running this example.")

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

# =============================================================================
# Example 1 — SimpleExecutor
# =============================================================================

print("=" * 60)
print("Example 1: SimpleExecutor")
print("=" * 60)

ex = SimpleExecutor(
    api_key=API_KEY,
    model=MODEL,
    system_prompt="You are a concise assistant. Keep answers to one sentence.",
)

result = ex.execute("What is the capital of Japan?")
print("Response:", result.content)
print("Rounds:", result.rounds)
print()

# =============================================================================
# Example 2 — SimpleExecutor with executor-level hooks
# =============================================================================

print("=" * 60)
print("Example 2: SimpleExecutor with executor hooks")
print("=" * 60)

hook_log: list[str] = []


def logging_hook(stage: str, content: str) -> None:
    """Log executor stage transitions."""
    preview = content[:60].replace("\n", " ")
    hook_log.append(f"[{stage}] {preview}...")
    return None  # don't modify content


def uppercase_output_hook(stage: str, content: str) -> str | None:
    """Upper-case the executor output for demo purposes."""
    if stage == "post_executor":
        return content.upper()
    return None


hooked_ex = SimpleExecutor(
    api_key=API_KEY,
    model=MODEL,
    system_prompt="Answer in exactly one short sentence.",
    hooks=[logging_hook, uppercase_output_hook],
)

result = hooked_ex.execute("What is 12 × 12?")
print("Response (upper-cased by post hook):", result.content)
print("Hook log:", hook_log)
print()

# =============================================================================
# Example 2b — SimpleExecutor with on_status / sinks
# =============================================================================

print("=" * 60)
print("Example 2b: SimpleExecutor with on_status and sinks")
print("=" * 60)

status_log: list[str] = []


def on_status(event: ProcessEvent) -> None:
    cost = (
        f"${event.estimated_cost_usd:.6f}"
        if event.estimated_cost_usd is not None
        else "n/a"
    )
    status_log.append(f"{event.kind} round={event.round} cost={cost}")


def audit_sink(event: ProcessEvent) -> None:
    if event.kind == "llm_call_end":
        status_log.append(f"  [sink] model={event.model} tools={event.tool_call_count}")


observed_ex = SimpleExecutor(
    api_key=API_KEY,
    model=MODEL,
    system_prompt="Reply in one word.",
    on_status=on_status,
    sinks=[audit_sink],
)

result = observed_ex.execute("Say hi.")
print("Response:", result.content)
print("Status log:")
for line in status_log:
    print(" ", line)
print()

# =============================================================================
# Example 3 — AgentExecutor
# =============================================================================

print("=" * 60)
print("Example 3: AgentExecutor")
print("=" * 60)


def get_temperature(city: str) -> dict:
    """Return the current temperature for a city."""
    data = {"london": 14, "tokyo": 22, "new york": 18, "sydney": 25}
    return {"city": city, "temperature_celsius": data.get(city.lower(), 20)}


spec = AgentSpec(
    name="WeatherAgent",
    persona="a friendly meteorologist",
    goals=["Provide accurate weather data for any city asked."],
    constraints=["Always include the unit (°C or °F) in your response."],
)

agent = Agent(spec, api_key=API_KEY, model=MODEL)
agent.register_tool(get_temperature)

agent_ex = AgentExecutor(agent=agent)

result = agent_ex.execute("What is the current temperature in Tokyo and London?")
print("Response:", result.content)
print("Tool rounds:", result.rounds)
print()

# =============================================================================
# Example 4 — AgentStructuredExecutor with a dataclass schema
# =============================================================================

print("=" * 60)
print("Example 4: AgentStructuredExecutor (dataclass schema)")
print("=" * 60)


@dataclasses.dataclass
class SentimentResult:
    label: str          # "positive" | "neutral" | "negative"
    confidence: float   # 0.0 – 1.0
    reasoning: str      # one-sentence explanation


sentiment_spec = AgentSpec(
    name="SentimentAnalyser",
    persona="an expert sentiment analysis model",
    constraints=["Respond ONLY with the JSON object — no extra text."],
)

sentiment_agent = Agent(sentiment_spec, api_key=API_KEY, model=MODEL)
sentiment_ex = AgentStructuredExecutor(
    agent=sentiment_agent,
    response_schema=SentimentResult,
)

outcome: StructuredOutcome[SentimentResult] = sentiment_ex.execute(
    "The new product is absolutely outstanding! I love it."
)
print("Raw JSON:", outcome.content)
print("Parsed:", outcome.structured)
assert outcome.structured is not None
print(f"  label={outcome.structured.label!r}")
print(f"  confidence={outcome.structured.confidence}")
print(f"  reasoning={outcome.structured.reasoning!r}")
print()

# =============================================================================
# Example 5 — AgentStructuredExecutor with a Pydantic model (if available)
# =============================================================================

print("=" * 60)
print("Example 5: AgentStructuredExecutor (Pydantic schema)")
print("=" * 60)

try:
    from pydantic import BaseModel

    class KeyFacts(BaseModel):
        topic: str
        facts: list[str]
        confidence: float

    facts_spec = AgentSpec(
        name="FactExtractor",
        persona="a precise fact-extraction engine",
        constraints=["Return ONLY a JSON object — no markdown, no prose."],
    )
    facts_agent = Agent(facts_spec, api_key=API_KEY, model=MODEL)
    facts_ex = AgentStructuredExecutor(agent=facts_agent, response_schema=KeyFacts)

    outcome = facts_ex.execute("Extract 3 key facts about the Python programming language.")
    print("Raw JSON:", outcome.content)
    parsed: KeyFacts = outcome.structured  # type: ignore
    print("Parsed:", parsed)
    print(f"  topic={parsed.topic!r}")
    for i, fact in enumerate(parsed.facts, 1):
        print(f"  fact {i}: {fact}")
    print()
except ImportError:
    print("pydantic not installed — skipping Pydantic example")
    print()

# =============================================================================
# Example 6 — Executor composability (sequential pipeline)
# =============================================================================

print("=" * 60)
print("Example 6: Sequential executor pipeline")
print("=" * 60)

# Stage 1: Extract key points
extractor_ex = SimpleExecutor(
    api_key=API_KEY,
    model=MODEL,
    system_prompt=(
        "You are a precise text summariser. "
        "Extract the 3 most important points as a numbered list."
    ),
)

# Stage 2: Translate the summary
translator_spec = AgentSpec(
    name="Translator",
    persona="a professional Spanish translator",
    goals=["Translate the provided text into fluent, natural Spanish."],
    constraints=["Output ONLY the translation — no preamble or explanation."],
)
translator_agent = Agent(translator_spec, api_key=API_KEY, model=MODEL)
translator_ex = AgentExecutor(agent=translator_agent)

source_text = (
    "Quantum computing uses quantum mechanical phenomena such as superposition "
    "and entanglement to perform calculations. Unlike classical bits, qubits can "
    "exist in multiple states simultaneously. This allows quantum computers to "
    "solve certain problems exponentially faster than classical computers."
)

print("Source text:", source_text[:80] + "…")
print()

summary = extractor_ex.execute(source_text)
print("Stage 1 — Key points:\n", summary.content)
print()

translation = translator_ex.execute(summary.content or "")
print("Stage 2 — Spanish translation:\n", translation.content)
print()

print("All executor examples completed successfully.")
