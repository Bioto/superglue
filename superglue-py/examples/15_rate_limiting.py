"""Example 15: Rate limiting — capping requests per second.

`requests_per_second` installs a token-bucket limiter shared across every
request made by the client (completions, tool calls, streaming). The limiter
ensures the throughput never exceeds the cap regardless of how many calls are
made concurrently.

This example fires 6 requests as fast as possible with a 2 req/s cap and
prints the elapsed time so you can see the limiter in action.

    OPENAI_API_KEY=sk-... uv run examples/15_rate_limiting.py
"""

import os
import time

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

# Cap this client at 2 requests per second.
# Useful when sharing a single API key across many callers or when the
# upstream provider imposes a QPS quota you want to stay safely under.
client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    requests_per_second=2,
)

questions = [
    "Name the planet closest to the Sun. One word.",
    "What colour is the sky on a clear day? One word.",
    "How many days are in a week? One word.",
    "What is the chemical symbol for water? One word.",
    "Name the largest ocean on Earth. Two words max.",
    "What is 7 multiplied by 8? One number.",
]

print(f"Sending {len(questions)} requests with requests_per_second=2 …\n")
start = time.monotonic()

for i, question in enumerate(questions):
    t0 = time.monotonic()
    result = client.complete(question)
    elapsed = time.monotonic() - t0
    total = time.monotonic() - start
    print(
        f"[{i + 1}/{len(questions)}] t={total:5.2f}s  "
        f"(+{elapsed:.2f}s)  →  {result.content}"
    )

total_elapsed = time.monotonic() - start
print(f"\nAll done in {total_elapsed:.2f}s.")
print(
    f"With a 2 req/s cap, 6 requests need at least "
    f"{(len(questions) - 1) / 2:.1f}s — limiter working if total ≥ that."
)
