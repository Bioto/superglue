/**
 * Example 14: Streaming vs buffered — latency comparison (TTFT).
 *
 *   OPENAI_API_KEY=sk-... node examples/14_streaming_vs_buffered.mjs
 */
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

const PROMPT =
  "Explain why Rust is considered a systems programming language.";

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY to run this example.");
    process.exit(1);
  }

  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    systemPrompt: "You are a helpful assistant. Always write at least 100 words.",
  });

  console.log("=== Buffered completion ===");
  let t0 = performance.now();
  const buffered = await client.complete(PROMPT);
  const bufferedTotal = (performance.now() - t0) / 1000;
  console.log(`Total time : ${bufferedTotal.toFixed(3)}s`);
  console.log(`Tokens     : ${JSON.stringify(buffered.usage)}`);
  console.log();

  console.log("=== Streaming completion ===");
  let firstTokenTime = null;
  t0 = performance.now();

  const streamResult = await client.stream(PROMPT, (_err, token) => {
    if (firstTokenTime === null) {
      firstTokenTime = (performance.now() - t0) / 1000;
    }
    process.stdout.write(token);
  });
  const streamingTotal = (performance.now() - t0) / 1000;

  console.log(`\n\nTime-to-first-token : ${firstTokenTime?.toFixed(3) ?? "n/a"}s`);
  console.log(`Total stream time   : ${streamingTotal.toFixed(3)}s`);
  console.log(`Tokens              : ${JSON.stringify(streamResult.usage)}`);
  console.log(
    `\nStreaming TTFT was ${(bufferedTotal - (firstTokenTime ?? 0)).toFixed(3)}s faster than buffered total.`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
