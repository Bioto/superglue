/**
 * Example 25: Model fallback chain.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const primary = process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini";
const backup = process.env.OPENAI_FALLBACK_MODEL ?? primary;

const client = createClient({
  apiKey,
  model: primary,
  modelFallbackModels: [primary, backup],
});

const result = await client.complete("Reply with exactly: fallback ok");
console.log("content:", result.content);
console.log("model_used:", result.model_used);
