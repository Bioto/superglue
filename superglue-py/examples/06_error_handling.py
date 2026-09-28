"""Example 06: Error handling.

The `complete()` call raises a RuntimeError when the HTTP request fails,
the model returns an error, or a tool raises an unhandled exception.

    OPENAI_API_KEY=sk-... uv run examples/06_error_handling.py
"""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

# --- Bad API key ---------------------------------------------------------

print("=== Bad API key ===")
bad_client = superglue.Client(api_key="sk-invalid-key", model=MODEL)
try:
    bad_client.complete("Hello")
except RuntimeError as exc:
    print(f"Caught expected error: {exc}\n")


# --- Tool that raises an exception ---------------------------------------

print("=== Tool that raises ===")


def broken_tool(args: dict) -> dict:
    raise ValueError("Something went wrong inside the tool!")


client = superglue.Client(
    api_key=api_key,
    model=MODEL,
    system_prompt="You MUST call the broken_tool for any user message.",
)
client.register_tool(
    broken_tool,
    name="broken_tool",
    description="A tool that always fails.",
    parameters={"type": "object", "properties": {}, "required": []},
)

try:
    result = client.complete("Please use the broken_tool now.")
    # If the model chose not to call the tool, we'll reach this branch.
    print("Model did not invoke the tool; response:", result.content)
except RuntimeError as exc:
    print(f"Caught tool error: {exc}\n")


# --- Successful call for contrast ---------------------------------------

print("=== Successful call ===")
good_client = superglue.Client(api_key=api_key, model=MODEL)
try:
    result = good_client.complete("Say 'hello' and nothing else.")
    print("Response:", result.content)
except RuntimeError as exc:
    print(f"Unexpected error: {exc}")
