"use strict";
/**
 * Ergonomic re-exports and helpers over the napi-generated [`index.js`](../index.js).
 *
 * Tools require explicit JSON Schema (`parameters`); pass an `execute` function that
 * receives and returns plain objects (or Promises).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.HookStage = exports.GuardrailStage = exports.version = exports.StatusEmitterJs = exports.Conversation = exports.Client = exports.AgentSpecJs = exports.AgentEngine = void 0;
exports.createClient = createClient;
exports.messageWithFileBytes = messageWithFileBytes;
exports.registerTool = registerTool;
exports.registerGuardrail = registerGuardrail;
const index_js_1 = require("../index.js");
Object.defineProperty(exports, "AgentEngine", { enumerable: true, get: function () { return index_js_1.AgentEngine; } });
Object.defineProperty(exports, "AgentSpecJs", { enumerable: true, get: function () { return index_js_1.AgentSpecJs; } });
Object.defineProperty(exports, "Client", { enumerable: true, get: function () { return index_js_1.Client; } });
Object.defineProperty(exports, "Conversation", { enumerable: true, get: function () { return index_js_1.Conversation; } });
Object.defineProperty(exports, "StatusEmitterJs", { enumerable: true, get: function () { return index_js_1.StatusEmitterJs; } });
Object.defineProperty(exports, "version", { enumerable: true, get: function () { return index_js_1.version; } });
exports.GuardrailStage = {
    INPUT: "input",
    OUTPUT: "output",
    BOTH: "both",
};
exports.HookStage = {
    PRE_COMPLETION: "pre_completion",
    POST_COMPLETION: "post_completion",
    PRE_TOOL: "pre_tool",
    POST_TOOL: "post_tool",
    ON_RETRY: "on_retry",
    PRE_BATCH_ITEM: "pre_batch_item",
    POST_BATCH_ITEM: "post_batch_item",
};
function envOpenAiModel() {
    const proc = globalThis
        .process;
    return proc?.env?.OPENAI_MODEL;
}
function createClient(options) {
    const model = options.model ?? envOpenAiModel();
    return new index_js_1.Client(options.apiKey, model, options.baseUrl, options.systemPrompt ?? undefined, options.maxToolRounds ?? undefined, options.maxRetries ?? undefined, options.retryInitialDelayMs ?? undefined, options.retryMaxDelayMs ?? undefined, options.retryMultiplier ?? undefined, options.requestsPerSecond ?? undefined, options.timeoutSecs ?? undefined, options.connectTimeoutSecs ?? undefined, options.maxOutputRetries ?? undefined, options.poolMaxIdlePerHost ?? undefined, options.poolIdleTimeoutSecs ?? undefined, options.reasoningEffort ?? undefined, options.statusEmitter ?? undefined, options.modelFallbackModels ?? undefined, options.apiKeys ?? undefined, options.requestsPerSecondFor ?? undefined, options.maxUploadBytes ?? undefined, options.toolMode ?? undefined, options.toolRouteModel ?? undefined, options.condenseToolMessages ?? undefined, options.aaakToolCondensing ?? undefined, options.summarizeContextEnabled ?? undefined, options.summarizeContextThreshold ?? undefined, options.summarizeContextKeepRecent ?? undefined, options.aaakCompressionEnabled ?? undefined, options.aaakCompressionModel ?? undefined);
}
/** Build a chat message JSON object with inline file bytes. */
function messageWithFileBytes(filename, fileBytes, text) {
    return index_js_1.Client.messageWithFileBytes(filename, fileBytes, text ?? undefined);
}
async function registerTool(client, def) {
    return client.registerTool(def.name, def.description, def.parameters, async (_err, args) => def.execute((args ?? {})), def.staticTool ?? undefined);
}
/** Normalize native callback args to `[stage, content]` strings. */
function guardrailStageContent(a, b) {
    if (typeof a === "string" && typeof b === "string") {
        return [a, b];
    }
    if (Array.isArray(a) && a.length >= 2) {
        return [String(a[0]), String(a[1])];
    }
    if (a && typeof a === "object" && !Array.isArray(a)) {
        const rec = a;
        if ("0" in rec && "1" in rec) {
            return [String(rec[0]), String(rec[1])];
        }
    }
    throw new Error("registerGuardrail: native callback did not yield (stage, content); check superglue-js / napi-rs version");
}
async function registerGuardrail(client, handler, stage, name) {
    // Must be a plain function (not `async`): an async wrapper always returns a Promise; napi then
    // treats rejections (e.g. `throw` in the handler) as unhandled instead of forwarding them to Rust.
    return client.registerGuardrail((_err, a, b) => {
        const [stageStr, contentStr] = guardrailStageContent(a, b);
        return handler(stageStr, contentStr);
    }, stage ?? undefined, name ?? undefined);
}
