"""Example 17 — Batch Completions.

Demonstrates three usage patterns:

1. Simple string list  — the quickest way to fire many prompts at once.
2. BatchRequest objects — attach per-item IDs and system-prompt overrides.
3. Error strategy       — choose how failures are handled (continue / skip / fail_fast).

Run:
    python examples/17_batch_completions.py
"""

from __future__ import annotations

import os
import time

import superglue
from superglue import BatchRequest

API_KEY = os.environ.get("OPENAI_API_KEY", "sk-...")
MODEL = os.environ.get("MODEL", "gpt-5.4-nano-2026-03-17-mini")

client = superglue.Client(
    api_key=API_KEY,
    model=MODEL,
    # Process up to 5 requests simultaneously (default).
    # Lower this if you hit rate limits:
    #   requests_per_second=2,
)

# ---------------------------------------------------------------------------
# 1. Simple string list
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Simple string list")
print("=" * 60)

questions = [
    "What is the capital of France?",
    "What is 7 × 8?",
    "Name one planet in our solar system.",
    "What colour is the sky on a clear day?",
]

t0 = time.monotonic()
response = client.batch(questions, max_concurrent=4)
elapsed = time.monotonic() - t0

print(f"\nBatch finished in {response.elapsed_secs:.2f}s "
      f"({response.successful}/{response.total_requests} succeeded)\n")

for result in response.results:
    status = "✓" if result.success else "✗"
    print(f"  {status} [{result.id}] {result.content or result.error}")

if response.total_usage:
    u = response.total_usage
    print(f"\n  Total tokens: {u['total_tokens']} "
          f"(prompt={u['prompt_tokens']}, completion={u['completion_tokens']})")

# ---------------------------------------------------------------------------
# 2. BatchRequest objects — per-item ID and system prompt override
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. BatchRequest with custom IDs and per-item system prompts")
print("=" * 60)

requests = [
    BatchRequest(
        prompt="Explain photosynthesis in one sentence.",
        id="biology-001",
        system_prompt="You are a primary school science teacher.",
    ),
    BatchRequest(
        prompt="Explain photosynthesis in one sentence.",
        id="biology-002",
        system_prompt="You are a PhD-level plant biologist.",
    ),
    BatchRequest(
        prompt="What is the boiling point of water?",
        id="chemistry-001",
        # No per-request system_prompt → uses the client-level default (none here).
    ),
]

response2 = client.batch(requests, max_concurrent=3)

print(f"\nBatch finished in {response2.elapsed_secs:.2f}s\n")
for result in response2.results:
    status = "✓" if result.success else "✗"
    print(f"  {status} [{result.id}] (rounds={result.rounds})")
    print(f"       {result.content or result.error}\n")

# ---------------------------------------------------------------------------
# 3. Error strategy — continue, skip, fail_fast
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. Error strategies")
print("=" * 60)

good_prompts = [BatchRequest(prompt="What is 1+1?", id=f"q{i}") for i in range(3)]

# --- continue (default): all results returned, failures flagged ---
print("\n  strategy='continue':")
resp_continue = client.batch(good_prompts, error_strategy="continue")
for r in resp_continue.results:
    flag = "ok" if r.success else "FAIL"
    print(f"    [{r.id}] {flag}: {r.content or r.error}")

# --- skip: failed items silently removed from results ---
print("\n  strategy='skip' (same prompts, no failures expected):")
resp_skip = client.batch(good_prompts, error_strategy="skip")
print(f"    returned {len(resp_skip.results)}/{resp_skip.total_requests} results")

print("\nDone.")
