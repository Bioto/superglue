"""Example 14: Streaming vs buffered — latency comparison.

Streaming returns the first token much faster than buffered completion, at
the cost of slightly more overhead overall. This example measures both and
prints the time-to-first-token (TTFT) for the streaming path.

    OPENAI_API_KEY=sk-... uv run examples/14_streaming_vs_buffered.py
"""

import os
import time

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt="You are a helpful assistant. Always write at least 100 words.",
)

PROMPT = "Explain why Rust is considered a systems programming language."

# --- Buffered ---------------------------------------------------------------
print("=== Buffered completion ===")
t0 = time.perf_counter()
result = client.complete(PROMPT)
buffered_total = time.perf_counter() - t0
print(f"Total time : {buffered_total:.3f}s")
print(f"Tokens     : {result.usage}")
print()

# --- Streaming --------------------------------------------------------------
print("=== Streaming completion ===")
first_token_time: float | None = None
t0 = time.perf_counter()


def on_token(token: str) -> None:
    global first_token_time
    if first_token_time is None:
        first_token_time = time.perf_counter() - t0
    print(token, end="", flush=True)


stream_result = client.stream(PROMPT, on_token=on_token)
streaming_total = time.perf_counter() - t0

print(f"\n\nTime-to-first-token : {first_token_time:.3f}s")
print(f"Total stream time   : {streaming_total:.3f}s")
print(f"Tokens              : {stream_result.usage}")
print(f"\nStreaming TTFT was {buffered_total - (first_token_time or 0):.3f}s faster than buffered total.")
