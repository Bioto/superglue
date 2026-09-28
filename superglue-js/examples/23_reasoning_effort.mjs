/**
 * Example 23: Per-agent reasoning effort (gluellm #361 parity).
 *
 *   OPENAI_API_KEY=sk-... node examples/23_reasoning_effort.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  AgentEngine,
  AgentSpecJs,
  createClient,
} = require("../dist/facade.js");

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run this example.");
  process.exit(1);
}

const model = process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini";

console.log("=".repeat(60));
console.log("1. Client-level reasoningEffort");
console.log("=".repeat(60));

const client = createClient({
  apiKey,
  model,
  reasoningEffort: "medium",
  systemPrompt: "Answer in one short sentence.",
});

const result = await client.response("What is 17 + 25?");
console.log("response():", result.content);
console.log();

console.log("=".repeat(60));
console.log("2. Per-call override");
console.log("=".repeat(60));

const high = await client.complete(
  "Name one benefit of higher reasoning effort.",
  undefined,
  undefined,
  undefined,
  "high",
);
console.log("complete(high):", high.content);
console.log();

console.log("=".repeat(60));
console.log("3. AgentSpecJs.reasoningEffort");
console.log("=".repeat(60));

const spec = new AgentSpecJs(
  "DeepThinker",
  "a careful analyst who thinks step-by-step",
  ["Give precise, well-reasoned answers"],
  null,
  model,
  16,
  3,
  null,
  "high",
);

const agent = new AgentEngine(spec, apiKey, undefined, model);
const agentResult = await agent.run(
  "Why might exponential backoff beat fixed-interval retry?",
);
console.log("Agent (spec reasoningEffort=high):", agentResult.content);
console.log();

console.log("All reasoning effort examples completed.");
