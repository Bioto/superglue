"""Example 02: Using a system prompt to control assistant behaviour.

The system prompt is injected as the first message of every conversation.

    OPENAI_API_KEY=sk-... uv run examples/02_system_prompt.py
"""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

# The system prompt shapes persona, tone, and constraints for every call made
# through this client instance.
client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt=(
        "You are a pirate captain. "
        "Always respond in pirate-speak, no matter what."
    ),
)

result = client.complete("What's the weather like today?")
print(result.content)
