/**
 * Example 06: Error handling (failed HTTP, model errors, tool failures).
 *
 *   OPENAI_API_KEY=sk-... node examples/06_error_handling.mjs
 *
 * Native HTTP pool may keep the process alive — `process.exit(0)` at the end.
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

  console.log("=== Bad API key ===");
  const badClient = createClient({
    apiKey: "sk-invalid-key",
    model: "gpt-5.4-nano-2026-03-17-mini",
  });
  try {
    await badClient.complete("Hello");
  } catch (exc) {
    console.log(`Caught expected error: ${exc}\n`);
  }

  console.log("=== Tool that throws ===");
  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    poolMaxIdlePerHost: 0,
    poolIdleTimeoutSecs: 1,
    systemPrompt:
      "Demo: you must call the tool named broken_tool for every user message.",
  });

  await registerTool(client, {
    name: "broken_tool",
    description: "A tool that always fails.",
    parameters: { type: "object", properties: {}, required: [] },
    execute: () => {
      throw new Error("Something went wrong inside the tool!");
    },
  });

  try {
    const result = await client.complete(
      "Invoke the broken_tool once (demo failure path).",
    );
    console.log("Model did not invoke the tool; response:", result.content);
  } catch (exc) {
    console.log(`Caught tool error (expected): ${exc}\n`);
  }

  console.log("=== Successful call ===");
  const goodClient = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    poolMaxIdlePerHost: 0,
    poolIdleTimeoutSecs: 1,
  });
  try {
    const result = await goodClient.complete("Say 'hello' and nothing else.");
    console.log("Response:", result.content);
  } catch (exc) {
    console.log(`Unexpected error: ${exc}`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
