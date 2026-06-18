/**
 * Ergonomic re-exports and helpers over the napi-generated [`index.js`](../index.js).
 *
 * Tools require explicit JSON Schema (`parameters`); pass an `execute` function that
 * receives and returns plain objects (or Promises).
 */

import {
  AgentEngine,
  AgentSpecJs,
  Client,
  Conversation,
  StatusEmitterJs,
  version,
} from "../index.js";

export {
  AgentEngine,
  AgentSpecJs,
  Client,
  Conversation,
  StatusEmitterJs,
  version,
};

export type {
  BatchRequestJs,
  BatchResponseJs,
  BatchResultJs,
  CompletionOutcomeJs,
  ResponseOutcomeJs,
  ResponseStreamOutcomeJs,
  StreamOutcomeJs,
} from "../index.js";

export const GuardrailStage = {
  INPUT: "input",
  OUTPUT: "output",
  BOTH: "both",
} as const;

export const HookStage = {
  PRE_COMPLETION: "pre_completion",
  POST_COMPLETION: "post_completion",
  PRE_TOOL: "pre_tool",
  POST_TOOL: "post_tool",
  ON_RETRY: "on_retry",
  PRE_BATCH_ITEM: "pre_batch_item",
  POST_BATCH_ITEM: "post_batch_item",
} as const;

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (
    args: Record<string, unknown>,
  ) => Record<string, unknown> | Promise<Record<string, unknown>>;
}

/** Constructor option bag matching the native `Client` constructor parameter order. */
export interface ClientOptions {
  apiKey: string;
  model?: string;
  baseUrl?: string;
  systemPrompt?: string | null;
  maxToolRounds?: number | null;
  maxRetries?: number | null;
  retryInitialDelayMs?: number | null;
  retryMaxDelayMs?: number | null;
  retryMultiplier?: number | null;
  requestsPerSecond?: number | null;
  timeoutSecs?: number | null;
  connectTimeoutSecs?: number | null;
  maxOutputRetries?: number | null;
  poolMaxIdlePerHost?: number | null;
  poolIdleTimeoutSecs?: number | null;
  reasoningEffort?: string | null;
  statusEmitter?: StatusEmitterJs | null;
  modelFallbackModels?: string[] | null;
  apiKeys?: Record<string, string> | null;
  requestsPerSecondFor?: Record<string, number> | null;
  maxUploadBytes?: number | null;
}

function envOpenAiModel(): string | undefined {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process;
  return proc?.env?.OPENAI_MODEL;
}

export function createClient(options: ClientOptions): Client {
  const model = options.model ?? envOpenAiModel();
  return new Client(
    options.apiKey,
    model,
    options.baseUrl,
    options.systemPrompt ?? undefined,
    options.maxToolRounds ?? undefined,
    options.maxRetries ?? undefined,
    options.retryInitialDelayMs ?? undefined,
    options.retryMaxDelayMs ?? undefined,
    options.retryMultiplier ?? undefined,
    options.requestsPerSecond ?? undefined,
    options.timeoutSecs ?? undefined,
    options.connectTimeoutSecs ?? undefined,
    options.maxOutputRetries ?? undefined,
    options.poolMaxIdlePerHost ?? undefined,
    options.poolIdleTimeoutSecs ?? undefined,
    options.reasoningEffort ?? undefined,
    options.statusEmitter ?? undefined,
    options.modelFallbackModels ?? undefined,
    options.apiKeys ?? undefined,
    options.requestsPerSecondFor ?? undefined,
    options.maxUploadBytes ?? undefined,
  );
}

/** Build a chat message JSON object with inline file bytes. */
export function messageWithFileBytes(
  filename: string,
  fileBytes: Uint8Array,
  text?: string | null,
): Record<string, unknown> {
  return Client.messageWithFileBytes(
    filename,
    fileBytes,
    text ?? undefined,
  ) as Record<string, unknown>;
}

export async function registerTool(
  client: Client,
  def: ToolDefinition,
): Promise<void> {
  return client.registerTool(
    def.name,
    def.description,
    def.parameters,
    // CalleeHandled threadsafe fn is invoked as (err, args). First arg is null on success.
    async (_err: unknown, args: unknown) =>
      def.execute((args ?? {}) as Record<string, unknown>),
  );
}

/**
 * Register a custom guardrail with an ergonomic `(stage, content) => …` handler.
 *
 * The native `ThreadsafeFunction` passes the Rust `(stage, content)` pair to JS either as
 * `(err, stage, content)` **or** as `(err, [stage, content])` depending on napi-rs tuple lowering.
 * Do not pass a raw `(stage, content)` two-arg function to `client.registerGuardrail` — use this helper.
 */
/** Sync only: native guardrail TSFN resolves to JSON `Value`, not a Promise (unlike `registerTool`). */
export type GuardrailHandlerFn = (stage: string, content: string) => string;

/** Normalize native callback args to `[stage, content]` strings. */
function guardrailStageContent(a: unknown, b: unknown): [string, string] {
  if (typeof a === "string" && typeof b === "string") {
    return [a, b];
  }
  if (Array.isArray(a) && a.length >= 2) {
    return [String(a[0]), String(a[1])];
  }
  if (a && typeof a === "object" && !Array.isArray(a)) {
    const rec = a as { 0?: unknown; 1?: unknown };
    if ("0" in rec && "1" in rec) {
      return [String(rec[0]), String(rec[1])];
    }
  }
  throw new Error(
    "registerGuardrail: native callback did not yield (stage, content); check superglue-js / napi-rs version",
  );
}

export async function registerGuardrail(
  client: Client,
  handler: GuardrailHandlerFn,
  stage?: string | null,
  name?: string | null,
): Promise<void> {
  // Must be a plain function (not `async`): an async wrapper always returns a Promise; napi then
  // treats rejections (e.g. `throw` in the handler) as unhandled instead of forwarding them to Rust.
  return client.registerGuardrail(
    (_err: unknown, a: unknown, b: unknown) => {
      const [stageStr, contentStr] = guardrailStageContent(a, b);
      return handler(stageStr, contentStr);
    },
    stage ?? undefined,
    name ?? undefined,
  );
}
