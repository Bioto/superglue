"""Example 09: Running multiple completions concurrently.

Because `complete()` releases the GIL (or runs GIL-free on CPython 3.14t),
threads can overlap their blocking HTTP calls, giving true parallelism from
Python without asyncio.

    OPENAI_API_KEY=sk-... uv run examples/09_concurrent_completions.py
"""

import os
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

prompts = [
    "Name the tallest mountain on Earth.",
    "What year did the Berlin Wall fall?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the speed of light in m/s?",
    "Name the first programming language.",
]


def ask(prompt: str) -> tuple[str, str | None]:
    result = client.complete(prompt)
    return prompt, result.content


start = time.perf_counter()

with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
    futures = {pool.submit(ask, p): p for p in prompts}
    for future in as_completed(futures):
        prompt, answer = future.result()
        print(f"Q: {prompt}")
        print(f"A: {answer}\n")

elapsed = time.perf_counter() - start
print(f"All {len(prompts)} completions finished in {elapsed:.2f}s (concurrent).")
