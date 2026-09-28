"""Demo: run after building the extension inside superglue-py/.

Without a real API key, the client construction works but complete() will fail.
Set OPENAI_API_KEY to run a live call.
"""

import os
from typing import Annotated

import superglue

print("superglue.version():", superglue.version())


def get_weather(location: Annotated[str, "City name, e.g. 'Paris'"]) -> dict:
    """Get the current weather for a location."""
    return {"location": location, "temp": 22, "unit": "C"}


api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print(
        "\nOPENAI_API_KEY not set — demonstrating client construction only.\n"
        "Set the env var and re-run for a live completion."
    )
    client = superglue.Client(
        api_key="sk-placeholder",
        model="gpt-5.4-nano-2026-03-17-mini",
        system_prompt="You are a concise assistant.",
    )
    # Schema is inferred automatically from annotations and docstring.
    client.register_tool(get_weather)
    print("Client and tool registered successfully (no network call made).")
else:
    client = superglue.Client(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
        system_prompt="You are a concise assistant. Use tools when relevant.",
    )
    client.register_tool(get_weather)

    result = client.complete("What is the weather in Paris?")
    print("\ncontent:", result.content)
    print("rounds:", result.rounds)
    print("usage:", result.usage)
