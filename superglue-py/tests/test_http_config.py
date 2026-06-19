"""HTTP client default configuration tests."""

from __future__ import annotations

import inspect

import superglue


def test_client_default_connect_timeout() -> None:
    sig = inspect.signature(superglue.Client.__init__)
    assert sig.parameters["connect_timeout_secs"].default == 30


def test_client_default_pool_max_idle_per_host() -> None:
    sig = inspect.signature(superglue.Client.__init__)
    assert sig.parameters["pool_max_idle_per_host"].default == 50


def test_glue_llm_embed_client_reused() -> None:
    llm = superglue.GlueLLM(api_key="sk-test")
    first = llm._embed_client()
    second = llm._embed_client()
    assert first is second
