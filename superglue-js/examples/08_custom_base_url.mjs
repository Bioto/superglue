/**
 * Example 08: Custom base URL — OpenAI-compatible providers.
 *
 *   # Ollama locally (defaults below):
 *   node examples/08_custom_base_url.mjs
 *
 *   # Groq:
 *   OPENAI_API_KEY=$GROQ_API_KEY OPENAI_BASE_URL=https://api.groq.com/openai \\
 *     OPENAI_MODEL=llama3-8b-8192 node examples/08_custom_base_url.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

async function main() {
  const apiKey = process.env.OPENAI_API_KEY ?? "ollama";
  const baseUrl =
    process.env.OPENAI_BASE_URL ??
    (apiKey.startsWith("sk-") ? "https://api.openai.com" : "http://localhost:11434");
  const model =
    process.env.OPENAI_MODEL ??
    (apiKey.startsWith("sk-") ? "gpt-4o-mini" : "llama3.2");

  console.log(`Using model '${model}' at ${baseUrl}\n`);

  const client = createClient({
    apiKey,
    model,
    baseUrl,
    systemPrompt: "You are a helpful assistant.",
  });

  try {
    const result = await client.complete("What is 7 times 8?");
    console.log("content:", result.content);
    console.log("usage:", result.usage);
  } catch (exc) {
    console.log(`Error (is the provider running?): ${exc}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
