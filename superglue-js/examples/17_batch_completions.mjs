/**
 * Expects OPENAI_API_KEY. Runs a small batch of prompts.
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
  const res = await client.batch(
    [
      { prompt: "Say: one." },
      { prompt: "Say: two." },
      { prompt: "Say: three." },
    ],
    2,
    "continue",
  );
  console.log(
    `batch ok=${res.successful} fail=${res.failed} elapsed=${res.elapsedSecs}s`,
  );
  for (const r of res.results) {
    console.log(r.id, r.success, r.content?.slice(0, 80));
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
