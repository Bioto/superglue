"""Example 24: HTTP connection pool and timeout defaults (gluellm #391 parity).

Default ``Client`` / ``Agent`` settings (aligned with gluellm):

- ``connect_timeout_secs=30`` — TCP connect timeout
- ``pool_max_idle_per_host=50`` — idle keep-alive connections per host
- ``timeout_secs=60`` — total request timeout (connect + response body)

Chat completions use the Rust ``reqwest`` pool on :class:`Client`.
:class:`GlueLLM.embed` uses a separate shared ``httpx`` client with
``max_connections=50`` and split connect (30 s) / read (120 s) timeouts.

Note: reqwest has no httpx-style ``pool_timeout`` (time waiting to acquire a
connection from the pool). Raise ``pool_max_idle_per_host`` for high fan-out.

    OPENAI_API_KEY=sk-... uv run examples/24_connection_pool.py
"""

import asyncio
import inspect
import os

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run the live sections.")
    raise SystemExit(1)

sig = inspect.signature(superglue.Client.__init__)
print("Client default connection settings:")
print(f"  connect_timeout_secs = {sig.parameters['connect_timeout_secs'].default}")
print(f"  pool_max_idle_per_host = {sig.parameters['pool_max_idle_per_host'].default}")
print(f"  timeout_secs = {sig.parameters['timeout_secs'].default}")
print()

# High fan-out: explicit pool sizing (same knobs as gluellm httpx limits)
pooled_client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    connect_timeout_secs=30,
    pool_max_idle_per_host=50,
    timeout_secs=90,
)

result = pooled_client.complete("Say 'pool ok' in two words.")
print("Chat completion:", result.content)
print()

print("GlueLLM.embed — shared httpx pool (async):")


async def embed_demo() -> None:
    async with superglue.GlueLLM(api_key=api_key) as llm:
        r1 = await llm.embed("hello world")
        r2 = await llm.embed(["foo", "bar"])
        assert llm._embed_client() is llm._embed_client()
        print(f"  batch 1 dim={len(r1.embeddings[0])}")
        print(f"  batch 2 count={len(r2.embeddings)}")


asyncio.run(embed_demo())
print("\nConnection pool example completed.")
