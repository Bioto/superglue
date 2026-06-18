"""Example 26: stream_response via the OpenAI Responses API."""

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

tokens = []


def on_token(delta: str) -> None:
    tokens.append(delta)


outcome = client.stream_response(
    "Count from 1 to 5 separated by spaces.",
    on_token=on_token,
)
print("deltas:", tokens)
print("content:", outcome.content)
print("id:", outcome.id)
print("model_used:", outcome.model_used)
