/**
 * Example 18: Lifecycle hooks (observe / mutate pipeline).
 *
 *   OPENAI_API_KEY=sk-... node examples/18_hooks.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  createClient,
  registerTool,
  HookStage,
} = require("../dist/facade.js");

const API_KEY = process.env.OPENAI_API_KEY;
if (!API_KEY) {
  console.error("OPENAI_API_KEY is not set — set it and re-run.");
  process.exit(1);
}

async function exampleObservationHooks() {
  console.log("\n=== Example 1: Observation hooks (logging) ===");

  const client = createClient({ apiKey: API_KEY });

  await client.registerHook(
    HookStage.PRE_COMPLETION,
    (ctx) => {
      console.log(
        `  → PRE_COMPLETION : ${JSON.stringify(String(ctx.content ?? "").slice(0, 60))}`,
      );
      return null;
    },
    "log-pre",
  );

  await client.registerHook(
    HookStage.POST_COMPLETION,
    (ctx) => {
      console.log(
        `  ← POST_COMPLETION: ${JSON.stringify(String(ctx.content ?? "").slice(0, 60))}`,
      );
      return null;
    },
    "log-post",
  );

  const result = await client.complete(
    "In one sentence, what is the speed of light?",
  );
  console.log(`  content: ${JSON.stringify(result.content)}`);
}

async function examplePreToolMutation() {
  console.log("\n=== Example 2: PreTool hook — mutate args before tool call ===");

  const client = createClient({ apiKey: API_KEY });

  await registerTool(client, {
    name: "get_weather",
    description: "Return the current temperature for a city.",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name" },
        unit: { type: "string", description: "celsius or fahrenheit" },
      },
      required: ["city"],
    },
    execute: (args) => {
      const city = String(args.city ?? "");
      const unit = String(args.unit ?? "celsius");
      const temps = { london: 15, paris: 18, "new york": 22, tokyo: 25 };
      const temp = temps[city.toLowerCase()] ?? 20;
      return { city, temperature: temp, unit };
    },
  });

  await client.registerHook(
    HookStage.PRE_TOOL,
    (ctx) => {
      const args = JSON.parse(String(ctx.content));
      args.unit = "fahrenheit";
      const tool = ctx.metadata?.tool_name;
      console.log(
        `  [pre_tool] Overriding unit → fahrenheit  (tool: ${tool ?? ""})`,
      );
      return JSON.stringify(args);
    },
    "force-fahrenheit",
  );

  const result = await client.complete(
    "What is the temperature in London in celsius?",
  );
  console.log(`  content: ${JSON.stringify(result.content)}`);
}

async function examplePostToolMutation() {
  console.log(
    "\n=== Example 3: PostTool hook — augment result before model sees it ===",
  );

  const client = createClient({ apiKey: API_KEY });

  await registerTool(client, {
    name: "lookup_user",
    description: "Look up a user by ID.",
    parameters: {
      type: "object",
      properties: {
        user_id: { type: "string", description: "User ID to look up" },
      },
      required: ["user_id"],
    },
    execute: ({ user_id }) => ({
      user_id,
      name: "Alice",
      role: "admin",
    }),
  });

  await client.registerHook(
    HookStage.POST_TOOL,
    (ctx) => {
      const result = JSON.parse(String(ctx.content));
      result._fetched_at = "2026-04-12T00:00:00Z";
      result._source = "internal-db";
      console.log("  [post_tool] Injected metadata into result");
      return JSON.stringify(result);
    },
    "inject-meta",
  );

  const result = await client.complete(
    "Look up user 'u-42' and tell me their role.",
  );
  console.log(`  content: ${JSON.stringify(result.content)}`);
}

async function exampleAbortOnError() {
  console.log("\n=== Example 4: Hook with error_strategy='abort' ===");

  const client = createClient({ apiKey: API_KEY });

  await client.registerHook(
    HookStage.PRE_COMPLETION,
    () => {
      throw new Error("Simulated hook failure");
    },
    "always-fail",
    "abort",
  );

  try {
    await client.complete("Hello!");
    console.log("  (should not reach here)");
  } catch (exc) {
    console.log(`  Caught expected error: ${exc}`);
  }
}

async function exampleSkipOnError() {
  console.log("\n=== Example 5: Hook with error_strategy='skip' (default) ===");

  const client = createClient({ apiKey: API_KEY });

  await client.registerHook(
    HookStage.PRE_COMPLETION,
    () => {
      throw new Error("Transient failure — will be skipped");
    },
    "flaky",
  );

  const result = await client.complete("What is 1 + 1?");
  console.log(
    `  content: ${JSON.stringify(result.content)}  (call succeeded despite hook error)`,
  );
}

async function exampleBatchHooks() {
  console.log("\n=== Example 6: Batch hooks (PRE_BATCH_ITEM / POST_BATCH_ITEM) ===");

  const client = createClient({ apiKey: API_KEY });
  let preCount = 0;
  let postCount = 0;

  await client.registerHook(
    HookStage.PRE_BATCH_ITEM,
    (ctx) => {
      preCount += 1;
      console.log(
        `  [pre_batch_item #${preCount}] prompt=${JSON.stringify(String(ctx.content ?? "").slice(0, 40))}`,
      );
      return null;
    },
    "batch-pre",
  );

  await client.registerHook(
    HookStage.POST_BATCH_ITEM,
    (ctx) => {
      postCount += 1;
      console.log(
        `  [post_batch_item #${postCount}] reply=${JSON.stringify(String(ctx.content ?? "").slice(0, 40))}`,
      );
      return null;
    },
    "batch-post",
  );

  const resp = await client.batch(
    [
      { prompt: "What is the capital of France?" },
      { prompt: "What is the capital of Japan?" },
      { prompt: "What is the capital of Brazil?" },
    ],
    3,
    "continue",
  );
  console.log(`  ${resp.successful}/${resp.total_requests} succeeded`);
  console.log(`  pre fires=${preCount}  post fires=${postCount}`);
}

async function main() {
  await exampleObservationHooks();
  await examplePreToolMutation();
  await examplePostToolMutation();
  await exampleAbortOnError();
  await exampleSkipOnError();
  await exampleBatchHooks();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
