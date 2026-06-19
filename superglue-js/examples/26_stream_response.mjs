/**
 * Example 26: streamResponse via the Responses API.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const client = createClient({
  apiKey,
  model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
});

const tokens = [];
const outcome = await client.streamResponse(
  "Count from 1 to 5 separated by spaces.",
  (delta) => tokens.push(delta),
);

console.log("deltas:", tokens);
console.log("content:", outcome.content);
console.log("id:", outcome.id);
console.log("model_used:", outcome.model_used);
