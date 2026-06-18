"""Example 03: Tool calling — one tool, the model decides when to use it.

Type annotations and the docstring are used automatically to build the tool
schema. No manual JSON Schema required.

    OPENAI_API_KEY=sk-... uv run examples/03_single_tool.py
"""

import os
from typing import Annotated, Literal

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)


def get_weather(
    location: Annotated[str, "City name, e.g. 'Paris'"],
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> dict:
    """Get the current weather for a given location."""
    # Fake implementation — swap with a real API call in production.
    return {"location": location, "temperature": 22, "unit": unit, "condition": "sunny"}


client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt="You are a helpful weather assistant. Use tools to answer questions.",
)

# superglue infers name, description, and JSON Schema from the function.
client.register_tool(get_weather)

result = client.complete("What's the weather like in Tokyo right now?")

print("content:", result.content)
print("rounds:", result.rounds)   # typically 2: first call triggers tool, second synthesises answer
print("usage:", result.usage)
