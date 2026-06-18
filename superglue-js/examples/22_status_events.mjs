/**
 * Example 22: Process events via StatusEmitter (gluellm #351 parity).
 *
 *   OPENAI_API_KEY=sk-... node examples/22_status_events.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, StatusEmitterJs } = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const emitter = new StatusEmitterJs();

emitter.subscribe((_err, event) => {
  if (!event) return;
  const costStr =
    event.estimated_cost_usd != null
      ? `$${event.estimated_cost_usd.toFixed(6)}`
      : "n/a";
  let extra = "";
  if (event.error_type) {
    extra = ` error=${JSON.stringify(event.error_type)}`;
  } else if (event.tool_call_count) {
    extra = ` tools=${event.tool_call_count}`;
  }
  console.log(
    `[${event.kind}] model=${event.model} round=${event.round} cost=${costStr}${extra}`,
  );
});

const client = createClient({
  apiKey,
  model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
  statusEmitter: emitter,
});

const result = await client.complete("Say hello in one word.");
console.log("Response:", result.content);
