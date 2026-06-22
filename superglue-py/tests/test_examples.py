"""Run every examples/*.py script (tests execute the examples)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SKIP_MARKERS = (
    "Set OPENAI_API_KEY",
    "OPENAI_API_KEY is not set",
    "OPENAI_API_KEY not set",
)


def _example_ok(script: Path, result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 0:
        return True
    if os.environ.get("OPENAI_API_KEY"):
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 1 and any(m in combined for m in SKIP_MARKERS)


def test_all_examples_run() -> None:
    scripts = sorted(EXAMPLES_DIR.glob("*.py"))
    assert scripts, f"no examples found in {EXAMPLES_DIR}"
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=EXAMPLES_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        assert _example_ok(script, result), (
            f"{script.name} failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.live
def test_all_examples_run_live() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    scripts = sorted(EXAMPLES_DIR.glob("*.py"))
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=EXAMPLES_DIR.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{script.name} failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
