"""Example 28: MCP tools via stdio (optional — set MCP_RUN=1)."""

import os
import superglue

if os.environ.get("MCP_RUN") != "1":
    print("Skip MCP example (set MCP_RUN=1 and ensure npx is available)")
    raise SystemExit(0)

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)

client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
)

client.connect_mcp_stdio(
    "npx",
    ["-y", "@modelcontextprotocol/server-everything"],
    prefix="mcp",
)

result = client.complete(
    "Name one available tool with the mcp__ prefix. One line only."
)
print("content:", result.content)
