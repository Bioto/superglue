/**
 * Expects OPENAI_API_KEY. Run from package root: node examples/01_simple_completion.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { createClient } = require("../dist/facade.js");

async function main() {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.error("Set OPENAI_API_KEY");
    process.exit(1);
  }

  const client = createClient({ apiKey: key });
  const outcome = await client.complete(
    "Reply with exactly one word: ok.",
  );
  console.log(outcome.content);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
