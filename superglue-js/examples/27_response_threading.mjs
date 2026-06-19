/**
 * Example 27: completeResponse with tool loop + previous_response_id.
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
  model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
});

await registerTool(client, {
  name: "echo",
  description: "Echo JSON arguments",
  parameters: { type: "object" },
  execute: (args) => ({ echo: args ?? {} }),
});

const result = await client.completeResponse(
  "Call echo with {\"n\": 7} then summarize in one short sentence.",
);

console.log("content:", result.content);
console.log("rounds:", result.rounds);
console.log("id:", result.id);
