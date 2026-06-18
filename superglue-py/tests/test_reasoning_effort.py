"""Reasoning effort and AgentSpec forwarding tests."""

from __future__ import annotations

import superglue


def test_agent_spec_reasoning_effort_field() -> None:
    spec = superglue.AgentSpec(
        name="Reasoner",
        persona="a careful thinker",
        reasoning_effort="high",
    )
    assert spec.reasoning_effort == "high"


def test_simple_executor_forwards_reasoning_effort() -> None:
    ex = superglue.SimpleExecutor(
        api_key="sk-test",
        reasoning_effort="medium",
    )
    assert ex._client._rust is not None
