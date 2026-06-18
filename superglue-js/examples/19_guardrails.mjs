/**
 * Example 19: Guardrails (blocklist, max length, PII, custom handlers).
 *
 *   OPENAI_API_KEY=sk-... node examples/19_guardrails.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  createClient,
  registerGuardrail,
  GuardrailStage,
  HookStage,
} = require("../dist/facade.js");

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini";

async function main() {
  if (!API_KEY) {
    console.error("Set OPENAI_API_KEY to run this example.");
    process.exit(1);
  }

  console.log("\n──── 1. Blocklist guardrail (input block) ────");

  let client = createClient({ apiKey: API_KEY, model: MODEL });
  await client.addBlocklistGuardrail(
    ["profanity", String.raw`\bbadword\b`],
    "block",
    GuardrailStage.INPUT,
    "profanity-filter",
  );

  try {
    const result = await client.complete("Please say the word badword.");
    console.log(`Content: ${result.content}`);
  } catch (e) {
    console.log(`Blocked as expected: ${e.name}: ${e.message}`);
  }

  console.log("\n──── 2. Blocklist guardrail (output redact) ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });
  await client.addBlocklistGuardrail(
    [String.raw`\b\d{3}-\d{2}-\d{4}\b`],
    "redact",
    GuardrailStage.OUTPUT,
    "ssn-redact",
  );

  try {
    const result = await client.complete(
      "Pretend your SSN is 123-45-6789 and mention it.",
    );
    console.log(`Content (redacted): ${result.content}`);
  } catch (e) {
    console.log(`Error: ${e}`);
  }

  console.log("\n──── 3. Max-length guardrail ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });
  await client.addMaxLengthGuardrail(50, 200, "truncate", "length-limits");

  try {
    const veryLongPrompt = "A".repeat(100);
    const result = await client.complete(veryLongPrompt);
    console.log(`Content (truncated): ${String(result.content).slice(0, 80)}...`);
  } catch (e) {
    console.log(`Blocked long input: ${e.name}: ${e.message}`);
  }

  console.log("\n──── 4. PII redaction guardrail ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });
  await client.addPiiGuardrail(GuardrailStage.OUTPUT, "pii-filter");

  try {
    const result = await client.complete(
      "Summarize: call me at 555-123-4567 or email me@example.com",
    );
    console.log(`Content (PII redacted): ${result.content}`);
  } catch (e) {
    console.log(`Error: ${e}`);
  }

  console.log("\n──── 5. Custom guardrail function ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });

  await registerGuardrail(
    client,
    (_stage, content) => {
      const competitors = ["competitor_a", "rival_corp", "other_llm"];
      for (const c of competitors) {
        if (content.toLowerCase().includes(c.toLowerCase())) {
          throw new Error(`competitor mention detected: '${c}'`);
        }
      }
      return content;
    },
    GuardrailStage.INPUT,
    "competitor-filter",
  );

  try {
    const result = await client.complete("Tell me about competitor_a.");
    console.log(`Content: ${result.content}`);
  } catch (e) {
    console.log(`Blocked by custom guardrail: ${e.name}: ${e.message}`);
  }

  try {
    const result = await client.complete(
      "Tell me about large language models in general.",
    );
    console.log(`Safe message — content: ${String(result.content).slice(0, 80)}`);
  } catch (e) {
    console.log(`Unexpected error: ${e}`);
  }

  console.log("\n──── 6. Custom output transform ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });

  const tokenPat = /\bsk-[a-zA-Z0-9]{20,}\b/g;
  await registerGuardrail(
    client,
    (_stage, content) => content.replace(tokenPat, "[API_KEY_REDACTED]"),
    GuardrailStage.OUTPUT,
    "api-key-redact",
  );

  try {
    const result = await client.complete(
      "Please echo the text: sk-abcdefghijklmnopqrstuvwxyz",
    );
    console.log(`Content (API key redacted): ${result.content}`);
  } catch (e) {
    console.log(`Error: ${e}`);
  }

  console.log("\n──── 7. Output guardrail with retry loop ────");

  let attemptCount = 0;
  client = createClient({
    apiKey: API_KEY,
    model: MODEL,
    maxOutputRetries: 3,
  });

  await registerGuardrail(
    client,
    (_stage, content) => {
      attemptCount += 1;
      if (content.length < 30 && attemptCount < 2) {
        throw new Error(
          `response too short (${content.length} chars), please elaborate`,
        );
      }
      return content;
    },
    GuardrailStage.OUTPUT,
    "quality-checker",
  );

  try {
    const result = await client.complete("Say 'hi' in exactly two letters.");
    console.log(
      `Content after retry: ${JSON.stringify(result.content)} (attempts: ${attemptCount})`,
    );
  } catch (e) {
    console.log(`All retries exhausted: ${e.name}: ${e.message}`);
  }

  console.log("\n──── 8. Guardrails and hooks together ────");

  client = createClient({ apiKey: API_KEY, model: MODEL });

  await client.registerHook(
    HookStage.PRE_COMPLETION,
    (ctx) => {
      console.log(
        `  [hook] PRE_COMPLETION — prompt=${JSON.stringify(String(ctx.content ?? "").slice(0, 60))}`,
      );
      return null;
    },
    "logger",
  );

  await registerGuardrail(
    client,
    (_stage, content) =>
      content.replace(/token=[A-Za-z0-9]+/g, "token=[REDACTED]"),
    GuardrailStage.INPUT,
    "token-redact",
  );

  try {
    const result = await client.complete(
      "My token=supersecret123, what should I do with it?",
    );
    console.log(`  Content: ${String(result.content).slice(0, 80)}`);
  } catch (e) {
    console.log(`  Error: ${e}`);
  }

  console.log("\nAll guardrail examples completed.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
