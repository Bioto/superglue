"""Example 08: Custom base URL — compatible providers.

Any OpenAI-compatible provider (Ollama, vLLM, Groq, Together AI, …) can be
used by pointing `base_url` at their endpoint and supplying the right model
name and API key.

    # Ollama running locally:
    uv run examples/08_custom_base_url.py

    # Groq:
    OPENAI_API_KEY=$GROQ_API_KEY OPENAI_BASE_URL=https://api.groq.com/openai \
        OPENAI_MODEL=llama3-8b-8192 uv run examples/08_custom_base_url.py
"""

import os
import superglue

# Defaults to a local Ollama instance when no OpenAI key is configured.
api_key = os.environ.get("OPENAI_API_KEY", "ollama")
base_url = os.environ.get("OPENAI_BASE_URL")
if base_url is None:
    base_url = (
        "https://api.openai.com"
        if api_key.startswith("sk-")
        else "http://localhost:11434"
    )
model = os.environ.get(
    "OPENAI_MODEL",
    "gpt-4o-mini" if api_key.startswith("sk-") else "llama3.2",
)

print(f"Using model '{model}' at {base_url}\n")

client = superglue.Client(
    api_key=api_key,
    model=model,
    base_url=base_url,
    system_prompt="You are a helpful assistant.",
)

try:
    result = client.complete("What is 7 times 8?")
    print("content:", result.content)
    print("usage:", result.usage)
except RuntimeError as exc:
    print(f"Error (is the provider running?): {exc}")
