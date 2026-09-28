/**
 * Demo: run after `npm run build` inside superglue-js/.
 *
 * Without a real API key, construction succeeds but complete() is skipped for the live branch.
 *   OPENAI_API_KEY=sk-... node examples/demo.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool, version } = require("../dist/facade.js");

async function main() {
  console.log("superglue-js version():", version());

  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    console.log(
      "\nOPENAI_API_KEY not set — demonstrating client construction only.\n" +
        "Set the env var and re-run for a live completion.",
    );
    const client = createClient({
      apiKey: "sk-placeholder",
      model: "gpt-5.4-nano-2026-03-17-mini",
      systemPrompt: "You are a concise assistant.",
    });
    await registerTool(client, {
      name: "get_weather",
      description: "Get the current weather for a location.",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "City name, e.g. 'Paris'" },
        },
        required: ["location"],
      },
      execute: ({ location }) => ({
        location,
        temp: 22,
        unit: "C",
      }),
    });
    console.log("Client and tool registered successfully (no network call made).");
    return;
  }

  const client = createClient({
    apiKey,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    baseUrl: process.env.OPENAI_BASE_URL ?? "https://api.openai.com",
    systemPrompt: "You are a concise assistant. Use tools when relevant.",
  });

  await registerTool(client, {
    name: "get_weather",
    description: "Get the current weather for a location.",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name, e.g. 'Paris'" },
      },
      required: ["location"],
    },
    execute: ({ location }) => ({
      location,
      temp: 22,
      unit: "C",
    }),
  });

  const result = await client.complete("What is the weather in Paris?");
  console.log("\ncontent:", result.content);
  console.log("rounds:", result.rounds);
  console.log("usage:", result.usage);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
