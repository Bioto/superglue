"""Example 12: Streaming with post-processing on the accumulated result.

This pattern is common in chatbot UIs: stream tokens to the screen in real
time for low perceived latency, then work with the full text once the stream
is done (e.g. to extract structured data, log the response, etc.).

    OPENAI_API_KEY=sk-... uv run examples/12_streaming_accumulate.py
"""

import os
import re

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt=(
        "You are a concise assistant. "
        "Always include at least one number in your responses."
    ),
)

# Tokens also accumulate inside the StreamOutcome, so we don't need to do it
# ourselves — but you can use the callback to update a UI widget etc.
token_count = 0


def on_token(token: str) -> None:
    global token_count
    token_count += 1
    print(token, end="", flush=True)


print("Streaming:\n")
result = client.stream(
    "List 5 interesting facts about the planet Mars. Use a numbered list.",
    on_token=on_token,
)
print("\n--- stream ended ---\n")

# Post-process the fully accumulated text.
numbers_found = re.findall(r"\d+", result.content)
print(f"Numbers mentioned : {numbers_found}")
print(f"Callback fired    : {token_count} times")
print(f"finish_reason     : {result.finish_reason}")
print(f"usage             : {result.usage}")
