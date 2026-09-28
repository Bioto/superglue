"""Example 04: Registering multiple tools.

The model picks among available tools on its own, and may call several of
them in a single conversation turn. All schemas are inferred automatically.

    OPENAI_API_KEY=sk-... uv run examples/04_multiple_tools.py
"""

import os
from typing import Annotated

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)


# --- Tool implementations ------------------------------------------------


def get_weather(location: Annotated[str, "City name, e.g. 'London'"]) -> dict:
    """Get the current weather for a location."""
    return {
        "location": location,
        "temperature": 18,
        "unit": "celsius",
        "condition": "partly cloudy",
    }


def search_web(query: Annotated[str, "Search query"]) -> dict:
    """Search the web for up-to-date information."""
    # Stub — replace with a real search API in production.
    return {
        "results": [
            {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
            {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"},
        ]
    }


def calculate(
    expression: Annotated[str, "A Python-style arithmetic expression, e.g. '2 ** 10'"],
) -> dict:
    """Evaluate a mathematical expression and return the result."""
    try:
        # WARNING: eval is used here purely for illustration. Never eval
        # untrusted input in a real application.
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


# --- Client setup --------------------------------------------------------

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt="You are a versatile assistant. Use tools whenever helpful.",
)

client.register_tool(get_weather)
client.register_tool(search_web)
client.register_tool(calculate)

# Ask something that might need several tools.
result = client.complete(
    "What's the weather in London, and what is 2 to the power of 16?"
)

print("content:", result.content)
print("rounds:", result.rounds)
print("usage:", result.usage)
