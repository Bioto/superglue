/**
 * Expects OPENAI_API_KEY. Streams tokens to stdout.
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
  process.stdout.write("STREAM: ");
  const out = await client.stream(
    "Count from 1 to 5 with commas.",
    (_err, delta) => {
      process.stdout.write(delta);
    },
  );
  process.stdout.write("\n");
  console.log("done", out.finishReason, out.requestId);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
