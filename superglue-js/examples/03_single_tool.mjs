/**
 * Expects OPENAI_API_KEY.
 * Run: node examples/03_single_tool.mjs
 *
 * The native HTTP client keeps idle connections open; without `process.exit(0)` at
 * the end, Node may not terminate until you press Ctrl+C.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient, registerTool } = require("../dist/facade.js");

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY");
    process.exit(1);
  }

  const client = createClient({
    apiKey: key,
    model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
    // Shrink idle pool so the process can exit sooner if something still holds the loop.
    poolMaxIdlePerHost: 0,
    poolIdleTimeoutSecs: 1,
    systemPrompt:
      "You are demonstrating the get_weather tool. It returns ILLUSTRATIVE stub JSON " +
      "(not a live forecast). After each tool call, reply in one short sentence that " +
      "quotes temperature, unit, and condition from the tool output. " +
      "Do not claim you cannot access data when the tool already returned JSON.",
  });

  await registerTool(client, {
    name: "get_weather",
    description:
      "Demo stub: returns illustrative weather fields for a city (not a real-time API). " +
      "Call this whenever the user asks about weather in the demo.",
    parameters: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "City name, e.g. 'Paris' or 'Tokyo'",
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit",
        },
      },
      required: ["location"],
    },
    execute: (args) => {
      const a = args ?? {};
      const location =
        typeof a.location === "string" && a.location.trim() !== ""
          ? a.location.trim()
          : "unknown";
      const unit = a.unit === "fahrenheit" ? "fahrenheit" : "celsius";
      // Avoid undefined fields: JSON omits them and the model may think data is missing.
      return {
        location,
        temperature: 22,
        unit,
        condition: "sunny",
        source: "example_stub",
        disclaimer: "Illustrative values for SDK demo only.",
      };
    },
  });

  const outcome = await client.complete(
    "What illustrative weather does the demo return for Tokyo?",
  );
  console.log("content:", outcome.content);
  console.log("rounds:", outcome.rounds);
  console.log("usage:", outcome.usage);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
