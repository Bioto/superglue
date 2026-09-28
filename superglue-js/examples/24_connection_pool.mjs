/**
 * Example 24: HTTP connection pool and timeout defaults (gluellm #391 parity).
 *
 * Default Client settings:
 *   connectTimeoutSecs = 30
 *   poolMaxIdlePerHost = 50
 *   timeoutSecs = 60
 *
 *   OPENAI_API_KEY=sk-... node examples/24_connection_pool.mjs
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { Client, createClient } = require("../dist/facade.js");

console.log("Client constructor arity (native binding):", Client.length);
console.log("Expected defaults: connectTimeoutSecs=30, poolMaxIdlePerHost=50, timeoutSecs=60");
console.log();

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) {
  console.error("Set OPENAI_API_KEY to run the live section.");
  process.exit(1);
}

const pooledClient = createClient({
  apiKey,
  model: process.env.OPENAI_MODEL ?? "gpt-5.4-nano-2026-03-17-mini",
  connectTimeoutSecs: 30,
  poolMaxIdlePerHost: 50,
  timeoutSecs: 90,
});

const result = await pooledClient.complete("Say 'pool ok' in two words.");
console.log("Chat completion:", result.content);
console.log("\nConnection pool example completed.");
