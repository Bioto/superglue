"""Example 29: multi-provider `provider:model` routing."""

import os
import superglue

openai_key = os.environ.get("OPENAI_API_KEY", "")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not openai_key:
    print("Set OPENAI_API_KEY to run OpenAI leg.")
    raise SystemExit(1)

api_keys = {"openai": openai_key}
if anthropic_key:
    api_keys["anthropic"] = anthropic_key

models = ["openai:gpt-4o-mini"]
if anthropic_key:
    models.append("anthropic:claude-sonnet-4-20250514")
else:
    print("ANTHROPIC_API_KEY not set — skipping anthropic model.")

for model in models:
    client = superglue.Client(
        api_key=openai_key,
        model=model,
        api_keys=api_keys,
    )
    out = client.complete("Say hello in one word.")
    print(f"{model}: {out.content}")
