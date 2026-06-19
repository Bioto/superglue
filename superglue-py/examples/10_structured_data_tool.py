"""Example 10: Returning rich structured data from tools.

Tools can return arbitrarily nested dicts. The model receives the full JSON
object and can reference individual fields in its response.

    OPENAI_API_KEY=sk-... uv run examples/10_structured_data_tool.py
"""

import os
from typing import Annotated

import superglue

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    print("Set OPENAI_API_KEY to run this example.")
    raise SystemExit(1)


def get_stock_quote(
    ticker: Annotated[str, "Stock ticker symbol, e.g. 'AAPL'"],
) -> dict:
    """Retrieve the latest stock quote for a ticker symbol."""
    ticker = ticker.upper()
    quotes = {
        "AAPL": {"price": 227.52, "change": +1.34, "change_pct": +0.59, "volume": 48_210_300},
        "MSFT": {"price": 415.80, "change": -2.10, "change_pct": -0.50, "volume": 22_100_000},
        "NVDA": {"price": 875.40, "change": +12.05, "change_pct": +1.40, "volume": 61_500_000},
    }
    if ticker not in quotes:
        return {"error": f"Ticker '{ticker}' not found"}
    return {"ticker": ticker, **quotes[ticker], "currency": "USD"}


def get_company_info(
    ticker: Annotated[str, "Stock ticker symbol, e.g. 'AAPL'"],
) -> dict:
    """Get basic information about a publicly listed company."""
    ticker = ticker.upper()
    info = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "employees": 161_000, "hq": "Cupertino, CA"},
        "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "employees": 221_000, "hq": "Redmond, WA"},
        "NVDA": {"name": "NVIDIA Corporation", "sector": "Semiconductors", "employees": 29_600, "hq": "Santa Clara, CA"},
    }
    if ticker not in info:
        return {"error": f"Ticker '{ticker}' not found"}
    return {"ticker": ticker, **info[ticker]}


client = superglue.Client(
    api_key=api_key,
    model=os.environ.get("OPENAI_MODEL", "gpt-5.4-nano-2026-03-17-mini"),
    system_prompt=(
        "You are a financial assistant. "
        "Always use the available tools to retrieve real data before answering."
    ),
)

client.register_tool(get_stock_quote)
client.register_tool(get_company_info)

result = client.complete(
    "Give me a brief summary of Apple (AAPL) — current price, today's change, "
    "and a couple of key company facts."
)

print(result.content)
print("\nrounds:", result.rounds)
print("usage:", result.usage)
