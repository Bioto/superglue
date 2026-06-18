"""Example 13: Concurrent streaming requests.

Because `stream()` releases the GIL (or runs GIL-free on CPython 3.14t),
multiple threads can stream simultaneously. Each thread prints its tokens
prefixed with an ID so you can see them interleaved.

    OPENAI_API_KEY=sk-... uv run examples/13_streaming_multiple_requests.py
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
)

PROMPTS = [
    ("A", "Write one sentence about the ocean."),
    ("B", "Write one sentence about mountains."),
    ("C", "Write one sentence about forests."),
]

print_lock = threading.Lock()


def stream_prompt(label: str, prompt: str) -> tuple[str, superglue.StreamOutcome]:
    def on_token(token: str) -> None:
        with print_lock:
            print(f"[{label}] {token}", end="", flush=True)

    result = client.stream(prompt, on_token=on_token)
    with print_lock:
        print()  # newline after each finished stream
    return label, result


start = time.perf_counter()
with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
    futures = [pool.submit(stream_prompt, lbl, msg) for lbl, msg in PROMPTS]
    results = {lbl: r for lbl, r in (f.result() for f in as_completed(futures))}

elapsed = time.perf_counter() - start
print(f"\nAll streams finished in {elapsed:.2f}s")

for label, outcome in sorted(results.items()):
    print(f"\n[{label}] finish_reason={outcome.finish_reason} usage={outcome.usage}")
