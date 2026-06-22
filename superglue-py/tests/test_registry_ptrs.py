"""Smoke tests for extension registry pointer exports."""

from __future__ import annotations

import superglue


def test_tools_registry_ptr_nonzero() -> None:
    client = superglue.Client(api_key="sk-test", model="gpt-5.4-nano-2026-03-17-mini")
    assert client.tools_registry_ptr() > 0


def test_hooks_registry_ptr_nonzero() -> None:
    client = superglue.Client(api_key="sk-test", model="gpt-5.4-nano-2026-03-17-mini")
    assert client.hooks_registry_ptr() > 0
