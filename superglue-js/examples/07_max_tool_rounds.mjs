/**
 * Example 07: Controlling max_tool_rounds (caps HTTP round-trips).
 *
 *   OPENAI_API_KEY=sk-... node examples/07_max_tool_rounds.mjs
 *
 * The model is steered to keep calling the tool until the engine hits `maxToolRounds`;
 * that rejection is the expected outcome (exit 0).
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool } = require("../dist/facade.js");

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY to run this example.");
    process.exit(1);
  }

  let callCount = 0;

  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    systemPrompt:
      "You are an agent that calls counter_tool repeatedly. " +
      "After each call check 'should_continue'; if true, call counter_tool again immediately. " +
      "Stop only when told otherwise.",
    maxToolRounds: 3,
  });

  await registerTool(client, {
    name: "counter_tool",
    description:
      "Increment a server-side counter and return the new count; signals whether to continue.",
    parameters: { type: "object", properties: {}, required: [] },
    execute: () => {
      callCount += 1;
      return { count: callCount, should_continue: true };
    },
  });

  try {
    const result = await client.complete(
      "Start counting. Keep going until you're told to stop.",
    );
    console.log("Final content:", result.content);
    console.log("Completion rounds:", result.rounds);
    console.log("Tool calls executed:", callCount);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("exceeded max tool rounds")) {
      console.log("Expected cap hit:", msg);
      console.log("Tool calls executed:", callCount);
    } else {
      throw e;
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
