"""Example 19: Guardrails

Guardrails protect your LLM pipeline by validating and optionally transforming
content at two points:

  - **Input** — checked on the user message before any LLM call.
    A blocked message raises immediately (no API request is made).
  - **Output** — checked on the assistant's final response.
    A blocked response triggers an LLM retry loop (up to ``max_output_retries``).

This file demonstrates:
  1. Built-in blocklist on input (block forbidden keywords)
  2. Built-in blocklist on output with redaction
  3. Built-in max-length guardrail (truncate)
  4. Built-in PII redaction
  5. Custom Python guardrail function (raises = block, returns str = transform)
  6. Output retry loop (custom evaluator rejects first attempt)

Run:
    export OPENAI_API_KEY=sk-...
    python 19_guardrails.py

Each example clearly prints what it tests and what the expected outcome is.
"""

from __future__ import annotations

import os
import re

import superglue
from superglue import Client, GuardrailStage

API_KEY = os.environ.get("OPENAI_API_KEY", "sk-test")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Built-in blocklist on INPUT — block profanity before the LLM is called
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 1. Blocklist guardrail (input block) ────")

client = Client(api_key=API_KEY, model=MODEL)
client.add_blocklist(
    ["profanity", r"\bbadword\b"],   # regex patterns
    action="block",                  # "block" (default) or "redact"
    stage=GuardrailStage.INPUT,      # INPUT, OUTPUT, or BOTH
    name="profanity-filter",
)

try:
    result = client.complete("Please say the word badword.")
    print(f"Content: {result.content}")
except Exception as e:
    print(f"Blocked as expected: {type(e).__name__}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Blocklist on OUTPUT with REDACT — PII in model responses is replaced
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 2. Blocklist guardrail (output redact) ────")

client = Client(api_key=API_KEY, model=MODEL)
client.add_blocklist(
    [r"\b\d{3}-\d{2}-\d{4}\b"],      # SSN-like pattern
    action="redact",                  # replace matches with [REDACTED]
    stage=GuardrailStage.OUTPUT,
    name="ssn-redact",
)

try:
    result = client.complete("Pretend your SSN is 123-45-6789 and mention it.")
    print(f"Content (redacted): {result.content}")
except Exception as e:
    print(f"Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Max-length guardrail — block long inputs, truncate long outputs
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 3. Max-length guardrail ────")

client = Client(api_key=API_KEY, model=MODEL)
client.add_max_length(
    max_input=50,            # block user messages > 50 chars
    max_output=200,          # truncate responses > 200 chars
    strategy="truncate",     # "block" raises, "truncate" silently shortens
    name="length-limits",
)

try:
    very_long_prompt = "A" * 100
    result = client.complete(very_long_prompt)
    print(f"Content (truncated): {result.content[:80]}...")
except Exception as e:
    print(f"Blocked long input: {type(e).__name__}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Built-in PII redaction — automatically redacts emails, phones, SSNs, cards
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 4. PII redaction guardrail ────")

client = Client(api_key=API_KEY, model=MODEL)
client.add_pii_guardrail(
    stage=GuardrailStage.OUTPUT,   # apply only to model responses
    name="pii-filter",
)

try:
    result = client.complete(
        "Summarize: call me at 555-123-4567 or email me@example.com"
    )
    print(f"Content (PII redacted): {result.content}")
except Exception as e:
    print(f"Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Custom Python guardrail — raise to block, return str to transform
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 5. Custom guardrail function ────")

client = Client(api_key=API_KEY, model=MODEL)

def no_competitor_mentions(content: str) -> str:
    """Block any content mentioning competitor names."""
    competitors = ["competitor_a", "rival_corp", "other_llm"]
    for c in competitors:
        if c.lower() in content.lower():
            raise ValueError(f"competitor mention detected: {c!r}")
    return content  # allow unchanged

client.register_guardrail(
    no_competitor_mentions,
    stage=GuardrailStage.INPUT,
    name="competitor-filter",
)

try:
    result = client.complete("Tell me about competitor_a.")
    print(f"Content: {result.content}")
except Exception as e:
    print(f"Blocked by custom guardrail: {type(e).__name__}: {e}")

# Safe message — should go through
try:
    result = client.complete("Tell me about large language models in general.")
    print(f"Safe message — content: {result.content[:80]}")
except Exception as e:
    print(f"Unexpected error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Custom output guardrail that transforms (redacts) content
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 6. Custom output transform ────")

client = Client(api_key=API_KEY, model=MODEL)

_TOKEN_PAT = re.compile(r"\bsk-[a-zA-Z0-9]{20,}\b")

def redact_api_keys(content: str) -> str:
    """Replace any API key patterns with [API_KEY_REDACTED]."""
    return _TOKEN_PAT.sub("[API_KEY_REDACTED]", content)

client.register_guardrail(
    redact_api_keys,
    stage=GuardrailStage.OUTPUT,
    name="api-key-redact",
)

try:
    result = client.complete("Please echo the text: sk-abcdefghijklmnopqrstuvwxyz")
    print(f"Content (API key redacted): {result.content}")
except Exception as e:
    print(f"Error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Output retry with max_output_retries
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 7. Output guardrail with retry loop ────")

# Pass max_output_retries=3 at construction time.
client = Client(api_key=API_KEY, model=MODEL, max_output_retries=3)

_attempt_count = 0

def quality_check(content: str) -> str:
    """Reject short responses; pass through long ones."""
    global _attempt_count
    _attempt_count += 1
    if len(content) < 30 and _attempt_count < 2:
        raise ValueError(f"response too short ({len(content)} chars), please elaborate")
    return content

client.register_guardrail(
    quality_check,
    stage=GuardrailStage.OUTPUT,
    name="quality-checker",
)

try:
    result = client.complete("Say 'hi' in exactly two letters.")
    print(f"Content after retry: {result.content!r} (attempts: {_attempt_count})")
except Exception as e:
    print(f"All retries exhausted: {type(e).__name__}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Guardrails + hooks together
# ──────────────────────────────────────────────────────────────────────────────

print("\n──── 8. Guardrails and hooks together ────")

from superglue import HookStage

client = Client(api_key=API_KEY, model=MODEL)

# Hook: observe what goes to the LLM
def log_pre(ctx: dict) -> None:
    print(f"  [hook] PRE_COMPLETION — prompt={ctx['content'][:60]!r}")

# Guardrail: redact secrets from the input
def redact_secrets(content: str) -> str:
    return re.sub(r"token=[A-Za-z0-9]+", "token=[REDACTED]", content)

client.register_hook(HookStage.PRE_COMPLETION, log_pre, name="logger")
client.register_guardrail(redact_secrets, stage=GuardrailStage.INPUT, name="token-redact")

try:
    result = client.complete("My token=supersecret123, what should I do with it?")
    print(f"  Content: {result.content[:80]}")
except Exception as e:
    print(f"  Error: {e}")

print("\nAll guardrail examples completed.")
