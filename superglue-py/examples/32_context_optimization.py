"""Example 32: GlueLLM-style context optimization (dynamic routing + condensing).

    OPENAI_API_KEY=sk-... uv run examples/32_context_optimization.py
"""

import os

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)


def get_weather(city: str) -> dict:
    """Get weather for a city."""
    return {"city": city, "temp_f": 72, "condition": "sunny"}


def calculate(expression: str) -> dict:
    """Evaluate a math expression."""
    return {"expression": expression, "result": 42}


def get_time() -> dict:
    """Return current UTC time (pinned static tool)."""
    return {"time": "12:00Z"}


client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt="You are a helpful assistant. Use tools when needed.",
    tool_mode="dynamic",
    condense_tool_messages=True,
)

client.register_tool(get_weather)
client.register_tool(calculate)
client.register_tool(get_time, static_tool=True)

result = client.complete("What's the weather in Paris?")
print("content:", result.content)
print("rounds:", result.rounds)

condensed = any(
    isinstance(m.get("content"), str) and "[Tool Results]" in m["content"]
    for m in (result.messages or [])
)
print("condensed tool round in history:", condensed)
