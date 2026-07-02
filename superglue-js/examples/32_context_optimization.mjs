/**
 * Example 32: GlueLLM-style context optimization (dynamic routing + condensing).
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const client = createClient({
  apiKey,
  model: process.env.OPENAI_MODEL ?? "openai:gpt-4o-mini",
  systemPrompt: "You are a helpful assistant. Use tools when needed.",
  toolMode: "dynamic",
  condenseToolMessages: true,
});

await registerTool(client, {
  name: "get_weather",
  description: "Get weather for a city",
  parameters: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  execute: ({ city }) => ({ city, temp_f: 72, condition: "sunny" }),
});

await registerTool(client, {
  name: "calculate",
  description: "Evaluate a math expression",
  parameters: {
    type: "object",
    properties: { expression: { type: "string" } },
    required: ["expression"],
  },
  execute: ({ expression }) => ({ expression, result: 42 }),
});

await registerTool(client, {
  name: "get_time",
  description: "Return current UTC time (pinned static tool)",
  parameters: { type: "object", properties: {} },
  execute: () => ({ time: "12:00Z" }),
  staticTool: true,
});

const result = await client.complete("What's the weather in Paris?");
console.log("content:", result.content);
console.log("rounds:", result.rounds);

const condensed = (result.messages ?? []).some(
  (m) => typeof m.content === "string" && m.content.includes("[Tool Results]"),
);
console.log("condensed tool round in history:", condensed);
