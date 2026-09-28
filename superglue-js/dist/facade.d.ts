/**
 * Ergonomic re-exports and helpers over the napi-generated [`index.js`](../index.js).
 *
 * Tools require explicit JSON Schema (`parameters`); pass an `execute` function that
 * receives and returns plain objects (or Promises).
 */
import { AgentEngine, AgentSpecJs, Client, Conversation, StatusEmitterJs, version } from "../index.js";
export { AgentEngine, AgentSpecJs, Client, Conversation, StatusEmitterJs, version, };
export type { BatchRequestJs, BatchResponseJs, BatchResultJs, CompletionOutcomeJs, ResponseOutcomeJs, ResponseStreamOutcomeJs, StreamOutcomeJs, } from "../index.js";
export declare const GuardrailStage: {
    readonly INPUT: "input";
    readonly OUTPUT: "output";
    readonly BOTH: "both";
};
export declare const HookStage: {
    readonly PRE_COMPLETION: "pre_completion";
    readonly POST_COMPLETION: "post_completion";
    readonly PRE_TOOL: "pre_tool";
    readonly POST_TOOL: "post_tool";
    readonly ON_RETRY: "on_retry";
    readonly PRE_BATCH_ITEM: "pre_batch_item";
    readonly POST_BATCH_ITEM: "post_batch_item";
};
export interface ToolDefinition {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    execute: (args: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>>;
    staticTool?: boolean;
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
    toolMode?: string | null;
    toolRouteModel?: string | null;
    condenseToolMessages?: boolean | null;
    aaakToolCondensing?: boolean | null;
    summarizeContextEnabled?: boolean | null;
    summarizeContextThreshold?: number | null;
    summarizeContextKeepRecent?: number | null;
    aaakCompressionEnabled?: boolean | null;
    aaakCompressionModel?: string | null;
}
export declare function createClient(options: ClientOptions): Client;
/** Build a chat message JSON object with inline file bytes. */
export declare function messageWithFileBytes(filename: string, fileBytes: Uint8Array, text?: string | null): Record<string, unknown>;
export declare function registerTool(client: Client, def: ToolDefinition): Promise<void>;
/**
 * Register a custom guardrail with an ergonomic `(stage, content) => …` handler.
 *
 * The native `ThreadsafeFunction` passes the Rust `(stage, content)` pair to JS either as
 * `(err, stage, content)` **or** as `(err, [stage, content])` depending on napi-rs tuple lowering.
 * Do not pass a raw `(stage, content)` two-arg function to `client.registerGuardrail` — use this helper.
 */
/** Sync only: native guardrail TSFN resolves to JSON `Value`, not a Promise (unlike `registerTool`). */
export type GuardrailHandlerFn = (stage: string, content: string) => string;
export declare function registerGuardrail(client: Client, handler: GuardrailHandlerFn, stage?: string | null, name?: string | null): Promise<void>;
//# sourceMappingURL=facade.d.ts.map