/**
 * Example 16: Retry configuration — backoff presets + live production call.
 *
 *   OPENAI_API_KEY=sk-... node examples/16_retry_configuration.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

function backoffSchedule(initialMs, multiplier, maxMs, retries) {
  const delays = [];
  for (let i = 0; i < retries; i++) {
    delays.push(Math.min(initialMs * multiplier ** i, maxMs) | 0);
  }
  return delays;
}

async function main() {
  const configs = [
    ["Default", 3, 1000, 6000, 3.0],
    ["Aggressive", 6, 100, 30_000, 2.0],
    ["Impatient", 1, 200, 200, 1.0],
    ["No retries", 0, 0, 0, 1.0],
  ];

  console.log("Backoff schedules (ms between each attempt):\n");
  console.log("Default connection settings: connectTimeoutSecs=30, poolMaxIdlePerHost=50\n");
  for (const [name, retries, initial, maxD, mult] of configs) {
    const schedule =
      retries > 0 ? backoffSchedule(initial, mult, maxD, retries) : [];
    const scheduleStr =
      schedule.length > 0 ? schedule.join(" → ") : "—";
    console.log(`  ${name.padEnd(12)} (${retries} retries): ${scheduleStr}`);
  }
  console.log();

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.log(
      "Set OPENAI_API_KEY to run a live completion with the production config.",
    );
    process.exit(0);
  }

  const productionClient = createClient({
    apiKey,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    maxRetries: 5,
    retryInitialDelayMs: 100,
    retryMaxDelayMs: 10_000,
    retryMultiplier: 2.0,
    requestsPerSecond: 8,
    timeoutSecs: 120,
    connectTimeoutSecs: 10,
  });

  const result = await productionClient.complete(
    "In one sentence, why is exponential backoff better than fixed-interval retry?",
  );
  console.log("Response:", result.content);
  console.log("Usage:", result.usage);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
