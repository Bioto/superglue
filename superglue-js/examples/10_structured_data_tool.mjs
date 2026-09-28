/**
 * Example 10: Rich structured data from tools (nested JSON for the model).
 *
 *   OPENAI_API_KEY=sk-... node examples/10_structured_data_tool.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool } = require("../dist/facade.js");

const QUOTES = {
  AAPL: {
    price: 227.52,
    change: 1.34,
    change_pct: 0.59,
    volume: 48_210_300,
  },
  MSFT: {
    price: 415.8,
    change: -2.1,
    change_pct: -0.5,
    volume: 22_100_000,
  },
  NVDA: {
    price: 875.4,
    change: 12.05,
    change_pct: 1.4,
    volume: 61_500_000,
  },
};

const INFO = {
  AAPL: {
    name: "Apple Inc.",
    sector: "Technology",
    employees: 161_000,
    hq: "Cupertino, CA",
  },
  MSFT: {
    name: "Microsoft Corporation",
    sector: "Technology",
    employees: 221_000,
    hq: "Redmond, WA",
  },
  NVDA: {
    name: "NVIDIA Corporation",
    sector: "Semiconductors",
    employees: 29_600,
    hq: "Santa Clara, CA",
  },
};

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY to run this example.");
    process.exit(1);
  }

  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    systemPrompt:
      "You are a financial assistant. " +
      "Always use the available tools to retrieve real data before answering.",
  });

  await registerTool(client, {
    name: "get_stock_quote",
    description: "Retrieve the latest stock quote for a ticker symbol.",
    parameters: {
      type: "object",
      properties: {
        ticker: { type: "string", description: "Stock ticker, e.g. 'AAPL'" },
      },
      required: ["ticker"],
    },
    execute: ({ ticker }) => {
      const t = String(ticker).toUpperCase();
      const q = QUOTES[t];
      if (!q) return { error: `Ticker '${t}' not found` };
      return { ticker: t, ...q, currency: "USD" };
    },
  });

  await registerTool(client, {
    name: "get_company_info",
    description: "Get basic information about a publicly listed company.",
    parameters: {
      type: "object",
      properties: {
        ticker: { type: "string", description: "Stock ticker, e.g. 'AAPL'" },
      },
      required: ["ticker"],
    },
    execute: ({ ticker }) => {
      const t = String(ticker).toUpperCase();
      const info = INFO[t];
      if (!info) return { error: `Ticker '${t}' not found` };
      return { ticker: t, ...info };
    },
  });

  const result = await client.complete(
    "Give me a brief summary of Apple (AAPL) — current price, today's change, " +
      "and a couple of key company facts.",
  );

  console.log(result.content);
  console.log("\nrounds:", result.rounds);
  console.log("usage:", result.usage);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
