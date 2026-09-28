"""Example 11: Basic streaming — print tokens as they arrive.

`client.stream(message, on_token=callback)` opens an SSE connection and calls
`on_token` with each content delta the model emits, then returns a
`StreamOutcome` with the full accumulated text, finish_reason, and usage.

    OPENAI_API_KEY=sk-... uv run examples/11_streaming.py
"""

import os
import sys

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt="You are a helpful assistant.",
)


def on_token(token: str) -> None:
    """Print each token immediately without a newline."""
    print(token, end="", flush=True)


print("Streaming response:\n")
result = client.stream(
    "Write a short poem about the Rust programming language.",
    on_token=on_token,
)

# After the stream ends, print a newline and the metadata.
print("\n")
print(f"finish_reason : {result.finish_reason}")
print(f"usage         : {result.usage}")
print(f"total chars   : {len(result.content)}")
