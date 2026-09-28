"""Example 16: Retry configuration — tuning backoff for production.

superglue retries transient failures automatically:
  - HTTP 429 (rate-limited by the provider)
  - HTTP 502 / 503 / 504 (gateway / service unavailable)
  - TCP-level timeouts and connection errors

The backoff is exponential:  delay = min(initial * multiplier^attempt, max)

Default connection pool settings (see also ``examples/24_connection_pool.py``):
  - connect_timeout_secs=30
  - pool_max_idle_per_host=50
  - timeout_secs=60

This example shows three common configurations and the formulas that describe
their backoff schedule.  No real API calls are made for the configuration
display; the live call at the end uses the recommended production settings.

    OPENAI_API_KEY=sk-... uv run examples/16_retry_configuration.py
"""

import os

import superglue

# ---------------------------------------------------------------------------
# Configuration presets (for illustration — pick one in real code)
# ---------------------------------------------------------------------------

# 1. Default — sensible for most interactive use
default_client = superglue.Client(
    api_key="sk-placeholder",   # no live call with this client
    model="gpt-5.4-nano-2026-03-17-mini",
    # max_retries=3, retry_initial_delay_ms=1000, retry_max_delay_ms=6000,
    # retry_multiplier=3.0  ← these are all the defaults
)

# 2. Aggressive — for high-throughput batch jobs that must succeed
aggressive_client = superglue.Client(
    api_key="sk-placeholder",
    model="gpt-5.4-nano-2026-03-17-mini",
    max_retries=6,
    retry_initial_delay_ms=100,
    retry_max_delay_ms=30_000,   # up to 30 s between attempts
    retry_multiplier=2.0,
)

# 3. Impatient — for latency-sensitive user-facing requests
impatient_client = superglue.Client(
    api_key="sk-placeholder",
    model="gpt-5.4-nano-2026-03-17-mini",
    max_retries=1,
    retry_initial_delay_ms=200,
    retry_max_delay_ms=200,      # flat, no growth
    retry_multiplier=1.0,
    timeout_secs=10,             # fail fast if the provider is slow
)

# 4. No retries — when the caller handles failure itself
no_retry_client = superglue.Client(
    api_key="sk-placeholder",
    model="gpt-5.4-nano-2026-03-17-mini",
    max_retries=0,
)


def backoff_schedule(initial_ms: int, multiplier: float, max_ms: int, retries: int) -> list[int]:
    delays = []
    for i in range(retries):
        d = min(initial_ms * (multiplier ** i), max_ms)
        delays.append(int(d))
    return delays


configs = [
    ("Default",     3,  50,  2_000, 2.0),
    ("Aggressive",  6, 100, 30_000, 2.0),
    ("Impatient",   1, 200,    200, 1.0),
    ("No retries",  0,   0,      0, 1.0),
]

print("Backoff schedules (ms between each attempt):\n")
for name, retries, initial, max_d, mult in configs:
    schedule = backoff_schedule(initial, mult, max_d, retries)
    schedule_str = " → ".join(f"{d:,}" for d in schedule) if schedule else "—"
    print(f"  {name:<12} ({retries} retries): {schedule_str}")

print()

# ---------------------------------------------------------------------------
# Live call with the recommended production configuration
# ---------------------------------------------------------------------------

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run a live completion with the production config.")
    raise SystemExit(0)

production_client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    # Retry up to 5 times with exponential backoff capped at 10 s
    max_retries=5,
    retry_initial_delay_ms=100,
    retry_max_delay_ms=10_000,
    retry_multiplier=2.0,
    # Stay safely under a 10 req/s org-level quota
    requests_per_second=8,
    # Generous timeout for long-running completions
    timeout_secs=120,
    connect_timeout_secs=30,   # default since gluellm #391 parity (was 15)
    pool_max_idle_per_host=50, # default idle keep-alive per host (was 8)
)

result = production_client.complete(
    "In one sentence, why is exponential backoff better than fixed-interval retry?"
)
print("Response:", result.content)
print("Usage:", result.usage)
