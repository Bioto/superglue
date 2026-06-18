/**
 * Example 02: Using a system prompt to control assistant behaviour.
 *
 *   OPENAI_API_KEY=sk-... node examples/02_system_prompt.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

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
      "You are a pirate captain. " +
      "Always respond in pirate-speak, no matter what.",
  });

  const result = await client.complete("What's the weather like today?");
  console.log(result.content);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
