/**
 * Example 13: Concurrent streaming requests (interleaved output).
 *
 *   OPENAI_API_KEY=sk-... node examples/13_streaming_multiple_requests.mjs
 */
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

const PROMPTS = [
  ["A", "Write one sentence about the ocean."],
  ["B", "Write one sentence about mountains."],
  ["C", "Write one sentence about forests."],
];

async function streamPrompt(client, label, prompt) {
  const out = await client.stream(prompt, (_err, token) => {
    process.stdout.write(`[${label}] ${token}`);
  });
  console.log();
  return [label, out];
}

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

  const start = performance.now();
  const settled = await Promise.all(
    PROMPTS.map(([label, msg]) => streamPrompt(client, label, msg)),
  );
  const elapsed = (performance.now() - start) / 1000;

  console.log(`\nAll streams finished in ${elapsed.toFixed(2)}s`);

  const byLabel = Object.fromEntries(settled);
  for (const label of ["A", "B", "C"]) {
    const outcome = byLabel[label];
    console.log(
      `\n[${label}] finish_reason=${outcome.finish_reason} usage=${JSON.stringify(outcome.usage)}`,
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
