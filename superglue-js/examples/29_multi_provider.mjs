/**
 * Example 29: multi-provider provider:model routing.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

const openaiKey = process.env.OPENAI_API_KEY;
if (!openaiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const apiKeys = { openai: openaiKey };
if (process.env.ANTHROPIC_API_KEY) {
  apiKeys.anthropic = process.env.ANTHROPIC_API_KEY;
} else {
  console.log("ANTHROPIC_API_KEY not set — skipping anthropic model.");
}

const models = ["openai:gpt-4o-mini"];
if (apiKeys.anthropic) {
  models.push("anthropic:claude-sonnet-4-20250514");
}

for (const model of models) {
  const client = createClient({ apiKey: openaiKey, model, apiKeys });
  const out = await client.complete("Say hello in one word.");
  console.log(`${model}: ${out.content}`);
}
