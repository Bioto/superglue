/**
 * Example 12: Streaming with post-processing on the accumulated result.
 *
 *   OPENAI_API_KEY=sk-... node examples/12_streaming_accumulate.mjs
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
      "You are a concise assistant. " +
      "Always include at least one number in your responses.",
  });

  let tokenCount = 0;

  console.log("Streaming:\n");
  const result = await client.stream(
    "List 5 interesting facts about the planet Mars. Use a numbered list.",
    (_err, token) => {
      tokenCount += 1;
      process.stdout.write(token);
    },
  );
  console.log("\n--- stream ended ---\n");

  const numbersFound = result.content.match(/\d+/g) ?? [];
  console.log(`Numbers mentioned : ${numbersFound.join(", ")}`);
  console.log(`Callback fired    : ${tokenCount} times`);
  console.log(`finish_reason     : ${result.finish_reason}`);
  console.log(`usage             : ${JSON.stringify(result.usage)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
