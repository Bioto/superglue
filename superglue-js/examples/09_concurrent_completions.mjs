/**
 * Example 09: Running multiple completions concurrently (Promise.all).
 *
 *   OPENAI_API_KEY=sk-... node examples/09_concurrent_completions.mjs
 */
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";

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
  });

  const prompts = [
    "Name the tallest mountain on Earth.",
    "What year did the Berlin Wall fall?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the speed of light in m/s?",
    "Name the first programming language.",
  ];

  const start = performance.now();
  const results = await Promise.all(
    prompts.map(async (prompt) => {
      const result = await client.complete(prompt);
      return { prompt, answer: result.content };
    }),
  );
  const elapsed = (performance.now() - start) / 1000;

  for (const { prompt, answer } of results) {
    console.log(`Q: ${prompt}`);
    console.log(`A: ${answer}\n`);
  }

  console.log(
    `All ${prompts.length} completions finished in ${elapsed.toFixed(2)}s (concurrent).`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
