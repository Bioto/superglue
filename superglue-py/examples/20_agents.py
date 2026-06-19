"""Example 20 — Agent system.

Demonstrates the superglue agent system: define an agent's persona, goals,
and constraints with ``AgentSpec``, then execute it with ``Agent`` or the
``Client.run_agent()`` convenience method.

Also covers ``AgentSpec.reasoning_effort`` for reasoning-capable models
(see ``examples/23_reasoning_effort.py`` for more detail).

The ``Agent`` class mirrors ``gluellm.AgentExecutor``:
- Compiles persona/goals/constraints into a structured system prompt
- Runs the full tool loop (up to ``max_tool_rounds``)
- Supports registered tools, lifecycle hooks, and guardrails
- Works independently from any ``Client`` instance

Prerequisites
-------------
Set ``OPENAI_API_KEY`` in your environment.

Run
---
    python examples/20_agents.py
"""

from __future__ import annotations

import os
import re
from typing import Annotated

from superglue import (
    Agent,
    AgentSpec,
    Client,
)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set OPENAI_API_KEY before running this example.")

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

# =============================================================================
# Example 1 — Minimal agent (persona only)
# =============================================================================

print("=" * 60)
print("Example 1: Minimal agent (persona only)")
print("=" * 60)

spec = AgentSpec(
    name="SimpleHelper",
    persona="a concise and helpful assistant",
)
print("Compiled system prompt:")
print(spec.compile_system_prompt())
print()

agent = Agent(spec, api_key=API_KEY, model=MODEL)
result = agent.run("What is 2 + 2? Answer in one sentence.")
print("Response:", result.content)
print()

# =============================================================================
# Example 2 — Agent with goals and constraints
# =============================================================================

print("=" * 60)
print("Example 2: Agent with goals and constraints")
print("=" * 60)

research_spec = AgentSpec(
    name="ResearchAssistant",
    persona="an expert research assistant specialising in science and technology",
    goals=[
        "Provide accurate, well-sourced answers backed by evidence.",
        "Explain complex concepts clearly for a general audience.",
        "Acknowledge the limits of your knowledge honestly.",
    ],
    constraints=[
        "Never speculate without clearly labelling it as speculation.",
        "Do not cite sources you cannot verify.",
        "Keep responses concise — prefer bullet points over long paragraphs.",
    ],
    model=MODEL,
    max_tool_rounds=4,
)

print("Compiled system prompt:")
print(research_spec.compile_system_prompt())
print()

researcher = Agent(research_spec, api_key=API_KEY)
result = researcher.run("Explain quantum entanglement in 3 bullet points.")
print("Response:", result.content)
print()

# =============================================================================
# Example 3 — Agent with tools registered
# =============================================================================

print("=" * 60)
print("Example 3: Agent with tools")
print("=" * 60)


def get_stock_price(
    ticker: Annotated[str, "Stock ticker symbol, e.g. 'AAPL'"],
) -> dict:
    """Get the latest stock price for a given ticker symbol."""
    prices = {"AAPL": 182.50, "MSFT": 415.20, "GOOGL": 175.80}
    price = prices.get(ticker.upper())
    if price is None:
        return {"error": f"Unknown ticker: {ticker}"}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


def get_company_info(
    name: Annotated[str, "Company name or partial name to search for"],
) -> dict:
    """Retrieve basic information about a publicly listed company."""
    db = {
        "apple": {"name": "Apple Inc.", "sector": "Technology", "employees": 164_000},
        "microsoft": {"name": "Microsoft Corp.", "sector": "Technology", "employees": 221_000},
    }
    key = name.lower()
    for k, v in db.items():
        if k in key or key in k:
            return v
    return {"error": f"No company found for: {name}"}


finance_spec = AgentSpec(
    name="FinanceAgent",
    persona="a sharp and data-driven financial analyst",
    goals=[
        "Provide real-time market data when asked.",
        "Give concise, actionable insights based on the data.",
    ],
    constraints=[
        "Never give investment advice — always say 'this is not financial advice'.",
        "Always cite the data source or tool used.",
    ],
)

finance_agent = Agent(finance_spec, api_key=API_KEY, model=MODEL)
finance_agent.register_tool(get_stock_price)
finance_agent.register_tool(get_company_info)

result = finance_agent.run(
    "What is Apple's current stock price and how many employees do they have?"
)
print("Response:", result.content)
print(f"Tool rounds: {result.rounds}")
print()

# =============================================================================
# Example 4 — Agent with lifecycle hooks
# =============================================================================

print("=" * 60)
print("Example 4: Agent with hooks")
print("=" * 60)

audit_log: list[dict] = []


def audit_hook(ctx: dict) -> None:
    """Log every pre-completion event to an in-memory audit trail."""
    audit_log.append({
        "stage": ctx.get("stage"),
        "content_preview": (ctx.get("content") or "")[:80],
    })
    return None  # None = keep content unchanged


def add_disclaimer(ctx: dict) -> str | None:
    """Append a safety disclaimer to every model output."""
    if ctx.get("stage") == "post_completion":
        content = ctx.get("content") or ""
        if not content.endswith("*"):
            return content + "\n\n*This information is for educational purposes only.*"
    return None


hooked_spec = AgentSpec(
    name="ComplianceAgent",
    persona="a compliance-aware business analyst",
    goals=["Answer business questions accurately"],
    constraints=["Always maintain a professional tone"],
)

hooked_agent = Agent(hooked_spec, api_key=API_KEY, model=MODEL)
hooked_agent.register_hook("pre_completion", audit_hook, name="audit")
hooked_agent.register_hook("post_completion", add_disclaimer, name="disclaimer")

result = hooked_agent.run("Summarise the top 3 risks of entering the EV market in 2 sentences.")
print("Response:", result.content)
print("Audit log entries:", len(audit_log), "—", audit_log)
print()

# =============================================================================
# Example 5 — Agent with guardrails
# =============================================================================

print("=" * 60)
print("Example 5: Agent with guardrails")
print("=" * 60)


def pii_guardrail(stage: str, content: str) -> str | None:
    """Block credit-card numbers on input; redact emails in output."""
    if stage == "input" and re.search(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", content):
        raise ValueError("Input contains a potential credit card number — blocked.")
    if stage == "output":
        return re.sub(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", "[REDACTED_EMAIL]", content)
    return None


safe_spec = AgentSpec(
    name="SafeAgent",
    persona="a privacy-conscious data assistant",
    goals=["Answer data-related questions safely"],
    constraints=["Never process or repeat personally identifiable information"],
    max_output_retries=2,
)

safe_agent = Agent(safe_spec, api_key=API_KEY, model=MODEL)
safe_agent.register_guardrail(pii_guardrail, stage="both", name="pii-guard")

# Normal query — passes guardrails
result = safe_agent.run("What is the best way to store user data securely?")
print("Safe response:", result.content)
print()

# PII in input — blocked by guardrail
try:
    safe_agent.run("My card number is 4111 1111 1111 1111. Is it valid?")
    print("ERROR: should have been blocked")
except Exception as e:
    print(f"Blocked as expected: {e}")
print()

# =============================================================================
# Example 6 — Sequential multi-agent pipeline (manager → worker)
# =============================================================================

print("=" * 60)
print("Example 6: Sequential multi-agent pipeline (manager → worker)")
print("=" * 60)

# Both agents run through the same Client, sharing the HTTP pool.
shared_client = Client(api_key=API_KEY, model=MODEL)

manager_spec = AgentSpec(
    name="ProjectManager",
    persona="a seasoned software project manager",
    goals=[
        "Break complex software tasks into 3–5 clear, actionable sub-tasks.",
        "Assign priorities (High / Medium / Low) to each sub-task.",
    ],
    constraints=[
        "Output exactly one numbered list — nothing else.",
        "Each sub-task must be completable in less than a day.",
    ],
)

developer_spec = AgentSpec(
    name="SeniorDeveloper",
    persona="a pragmatic senior software engineer who writes clean, idiomatic Python",
    goals=[
        "Implement the sub-task provided concisely.",
        "Include brief inline comments where the logic is non-obvious.",
    ],
    constraints=[
        "Write Python 3.12+ code only.",
        "Do not add placeholder TODO comments — either implement it or explain why not.",
    ],
)

task = "Build a rate-limited HTTP client wrapper in Python."

print(f"Task: {task}")
print()

plan_result = shared_client.run_agent(manager_spec, task)
print("Manager output (plan):")
print(plan_result.content)
print()

if plan_result.content:
    dev_prompt = f"Implement sub-task 1 from this plan:\n{plan_result.content}"
    code_result = shared_client.run_agent(developer_spec, dev_prompt)
    print("Developer output (code):")
    print(code_result.content)
    print()

# =============================================================================
# Example 7 — Verbatim system_prompt override
# =============================================================================

print("=" * 60)
print("Example 7: Custom system_prompt override")
print("=" * 60)

pirate_spec = AgentSpec(
    name="PirateBot",
    persona="ignored when system_prompt is set",
    system_prompt=(
        "You are a pirate. Respond exclusively in pirate speak. "
        "Ye shall never break character, arrr!"
    ),
)

print("Compiled prompt (override):")
print(pirate_spec.compile_system_prompt())
print()

pirate = Agent(pirate_spec, api_key=API_KEY, model=MODEL)
result = pirate.run("Hello! How are you today?")
print("Pirate response:", result.content)
print()

# =============================================================================
# Example 8 — Reasoning effort on AgentSpec
# =============================================================================

print("=" * 60)
print("Example 8: AgentSpec.reasoning_effort")
print("=" * 60)

reasoner_spec = AgentSpec(
    name="Reasoner",
    persona="a careful analyst",
    goals=["Provide well-reasoned, accurate answers"],
    constraints=["Keep answers to 2–3 sentences"],
    reasoning_effort="high",
    model=MODEL,
)

print(f"Agent reasoning_effort: {reasoner_spec.reasoning_effort!r}")
reasoner = Agent(reasoner_spec, api_key=API_KEY)
result = reasoner.run("What is one trade-off when increasing reasoning effort?")
print("Response:", result.content)
print()

print("All examples completed successfully.")
