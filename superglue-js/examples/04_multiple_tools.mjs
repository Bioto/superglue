/**
 * Example 04: Registering multiple tools (explicit JSON Schema + execute).
 *
 *   OPENAI_API_KEY=sk-... node examples/04_multiple_tools.mjs
 *
 * Same as 03: the native client can keep idle sockets open — we call `process.exit(0)`
 * so Node exits without Ctrl+C.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool } = require("../dist/facade.js");

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY to run this example.");
    process.exit(1);
  }

  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    poolMaxIdlePerHost: 0,
    poolIdleTimeoutSecs: 1,
    systemPrompt:
      "You are demonstrating registered tools. All tools return STUB / illustrative data " +
      "(not live web or real forecasts). You must call the appropriate tools and then answer " +
      "using only their JSON. Do not say you cannot fetch data when a tool already returned JSON.",
  });

  await registerTool(client, {
    name: "get_weather",
    description:
      "Demo stub: illustrative weather for a city (not a live API). Use for any weather question.",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name, e.g. 'London'" },
      },
      required: ["location"],
    },
    execute: (args) => {
      const a = args ?? {};
      const location =
        typeof a.location === "string" && a.location.trim() !== ""
          ? a.location.trim()
          : "unknown";
      return {
        location,
        temperature: 18,
        unit: "celsius",
        condition: "partly cloudy",
        disclaimer: "Illustrative values for SDK demo only.",
      };
    },
  });

  await registerTool(client, {
    name: "search_web",
    description:
      "Demo stub: returns fake search hits (not the real web). Use when a search-like answer is needed.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
      },
      required: ["query"],
    },
    execute: (args) => {
      const a = args ?? {};
      const query = typeof a.query === "string" ? a.query : "";
      return {
        results: [
          { title: `Result 1 for '${query}'`, url: "https://example.com/1" },
          { title: `Result 2 for '${query}'`, url: "https://example.com/2" },
        ],
        disclaimer: "Stub results for SDK demo only.",
      };
    },
  });

  await registerTool(client, {
    name: "calculate",
    description:
      "Evaluate a mathematical expression and return the result (demo / local eval).",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description: "Arithmetic expression, e.g. '2 ** 10'",
        },
      },
      required: ["expression"],
    },
    execute: (args) => {
      const a = args ?? {};
      const expression =
        typeof a.expression === "string" ? a.expression : String(a.expression ?? "");
      try {
        // eslint-disable-next-line no-new-func
        const result = Function(`"use strict"; return (${expression})`)();
        return { result };
      } catch (exc) {
        return { error: String(exc) };
      }
    },
  });

  const result = await client.complete(
    "Using the demo tools: (1) illustrative weather for London, (2) 2 to the power of 16 via calculate.",
  );

  console.log("content:", result.content);
  console.log("rounds:", result.rounds);
  console.log("usage:", result.usage);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
