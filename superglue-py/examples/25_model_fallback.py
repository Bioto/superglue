"""Example 25: Model fallback chain on the client."""

import os
import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

primary = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")
backup = os.environ.get("OPENAI_FALLBACK_MODEL", primary)

client = superglue.Client(
    api_key=api_key,
    model=primary,
    model_fallback_models=[primary, backup],
)

result = client.complete("Reply with exactly: fallback ok")
print("content:", result.content)
print("model_used:", result.model_used)
