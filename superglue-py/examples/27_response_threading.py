"""Example 27: Responses API tool loop with previous_response_id threading."""

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


def echo(args: dict) -> dict:
    return {"echo": args}


client.register_tool(
    echo,
    name="echo",
    description="Echo JSON arguments",
    parameters={"type": "object"},
)

result = client.complete_response(
    "Call echo with {\"n\": 7} then summarize in one short sentence."
)
print("content:", result.content)
print("rounds:", result.rounds)
print("id:", result.id)
