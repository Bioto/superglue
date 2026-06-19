"""Example 07: Controlling max_tool_rounds.

`max_tool_rounds` caps how many HTTP round-trips the engine makes before
returning, preventing runaway agentic loops. When the limit is reached the
latest assistant message is returned even if tools are still pending.

    OPENAI_API_KEY=sk-... uv run examples/07_max_tool_rounds.py
"""

import os

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

call_count = 0


def counter_tool() -> dict:
    """Increment a server-side counter and return the new count.

    Returns the current count and whether the agent should continue.
    """
    global call_count
    call_count += 1
    # Always ask the model to call again, creating an infinite loop without a cap.
    return {"count": call_count, "should_continue": True}


# Allow up to 3 tool invocations before the engine stops.
client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt=(
        "You are an agent that calls counter_tool repeatedly. "
        "After each call check 'should_continue'; if true, call counter_tool again immediately. "
        "Stop only when told otherwise."
    ),
    max_tool_rounds=3,
)

client.register_tool(counter_tool)

try:
    result = client.complete("Start counting. Keep going until you're told to stop.")
except RuntimeError as exc:
    if "max tool rounds" not in str(exc).lower():
        raise
    print("Stopped at max_tool_rounds (expected):", exc)
    print("Tool calls executed:", call_count)
    raise SystemExit(0)

print("Final content:", result.content)
print("Completion rounds:", result.rounds)
print("Tool calls executed:", call_count)
