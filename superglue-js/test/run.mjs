#!/usr/bin/env node
/** Smoke tests + execute every examples/*.mjs. */

import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { Client, version } = require("..");

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");
const examplesDir = join(root, "examples");

const SKIP_MARKERS = [
  "Set OPENAI_API_KEY",
  "OPENAI_API_KEY is not set",
  "OPENAI_API_KEY not set",
];

function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    process.exit(1);
  }
}

function exampleOk(script, result) {
  if (result.status === 0) return true;
  if (process.env.OPENAI_API_KEY) return false;
  const combined = `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
  return result.status === 1 && SKIP_MARKERS.some((m) => combined.includes(m));
}

function smokeTest() {
  assert(typeof version() === "string" && version().length > 0, "version()");

  const client = new Client("sk-test", "gpt-5.4-nano-2026-03-17-mini");
  assert(client.toolsRegistryPtr() > 0, "toolsRegistryPtr");
  assert(client.hooksRegistryPtr() > 0, "hooksRegistryPtr");
  console.log("smoke: ok");
}

function runExamples() {
  const scripts = readdirSync(examplesDir)
    .filter((f) => f.endsWith(".mjs"))
    .sort();
  assert(scripts.length > 0, "examples exist");

  for (const script of scripts) {
    const path = join(examplesDir, script);
    const result = spawnSync(process.execPath, [path], {
      cwd: root,
      encoding: "utf8",
      env: process.env,
    });
    if (!exampleOk(script, result)) {
      console.error(`example ${script} failed (exit ${result.status})`);
      console.error(result.stdout);
      console.error(result.stderr);
      process.exit(1);
    }
    console.log(`example ${script}: ok`);
  }
}

smokeTest();
runExamples();
console.log("all tests passed");
