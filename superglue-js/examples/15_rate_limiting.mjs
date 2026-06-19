/**
 * Example 15: Rate limiting — requests_per_second token bucket.
 *
 *   OPENAI_API_KEY=sk-... node examples/15_rate_limiting.mjs
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
    requestsPerSecond: 2,
  });

  const questions = [
    "Name the planet closest to the Sun. One word.",
    "What colour is the sky on a clear day? One word.",
    "How many days are in a week? One word.",
    "What is the chemical symbol for water? One word.",
    "Name the largest ocean on Earth. Two words max.",
    "What is 7 multiplied by 8? One number.",
  ];

  console.log(
    `Sending ${questions.length} requests with requestsPerSecond=2 …\n`,
  );
  const start = performance.now();

  for (let i = 0; i < questions.length; i++) {
    const question = questions[i];
    const t0 = performance.now();
    const result = await client.complete(question);
    const elapsed = (performance.now() - t0) / 1000;
    const total = (performance.now() - start) / 1000;
    console.log(
      `[${i + 1}/${questions.length}] t=${total.toFixed(2)}s  ` +
        `(+${elapsed.toFixed(2)}s)  →  ${result.content}`,
    );
  }

  const totalElapsed = (performance.now() - start) / 1000;
  console.log(`\nAll done in ${totalElapsed.toFixed(2)}s.`);
  console.log(
    `With a 2 req/s cap, 6 requests need at least ${(questions.length - 1) / 2}s — limiter working if total ≥ that.`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
