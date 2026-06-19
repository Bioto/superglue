"""Example 30: streaming completion with tool rounds."""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "openai:gpt-4o-mini"),
)


def echo(args: dict) -> dict:
    return {"echo": args}


client.register_tool(echo)

printed = []
result = client.stream(
    "Call echo with {\"msg\":\"hi\"} then say OK",
    on_token=lambda t: printed.append(t),
)
print("\ncontent:", result.content)
print("printed:", "".join(printed))
