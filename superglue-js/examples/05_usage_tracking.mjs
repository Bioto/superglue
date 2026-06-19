/**
 * Example 05: Token usage and cost tracking via StatusEmitter.
 *
 *   OPENAI_API_KEY=sk-... node examples/05_usage_tracking.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, StatusEmitterJs } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const model = process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini";

const emitter = new StatusEmitterJs();
const metrics = { costUsd: 0, calls: 0 };

function trackCost(_err, event) {
  if (!event) return;
  if (event.kind !== "llm_call_end") return;
  metrics.calls += 1;
  if (event.estimated_cost_usd != null) {
    metrics.costUsd += event.estimated_cost_usd;
  }
  const usage = event.usage ?? {};
  console.log(
    `  llm_call #${metrics.calls}: tokens=${usage.total_tokens ?? "?"} est=$${(event.estimated_cost_usd ?? 0).toFixed(6)}`,
  );
}

emitter.subscribe(trackCost);

const client = createClient({
  apiKey,
  model,
  statusEmitter: emitter,
});

const prompts = [
  "Name three programming languages invented before 1980.",
  "What is the difference between TCP and UDP? One sentence.",
  "Give me a haiku about Rust.",
];

let totalPrompt = 0;
let totalCompletion = 0;

for (const prompt of prompts) {
  const result = await client.complete(prompt);
  const usage = result.usage ?? {};
  const pt = usage.prompt_tokens ?? 0;
  const ct = usage.completion_tokens ?? 0;
  totalPrompt += pt;
  totalCompletion += ct;
  console.log(`Q: ${prompt}`);
  console.log(`A: ${result.content}`);
  console.log(`   tokens — prompt: ${pt}, completion: ${ct}\n`);
}

console.log(
  `Session totals: ${totalPrompt} prompt + ${totalCompletion} completion = ${totalPrompt + totalCompletion} tokens`,
);
console.log(`Estimated cost (StatusEmitter): $${metrics.costUsd.toFixed(6)}`);
