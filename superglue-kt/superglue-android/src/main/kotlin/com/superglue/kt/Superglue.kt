package com.superglue.kt

import kotlin.PublishedApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

private val configJson = Json { encodeDefaults = true }

/**
 * superglue version string (Rust crate `CARGO_PKG_VERSION`).
 */
public fun version(): String = SuperglueNativeJni.version()

public class SuperglueException(message: String) : RuntimeException(message)

// --- Constants (JNI stage strings) ---

public object HookStage {
    public const val PRE_COMPLETION: String = "pre_completion"
    public const val POST_COMPLETION: String = "post_completion"
    public const val PRE_TOOL: String = "pre_tool"
    public const val POST_TOOL: String = "post_tool"
    public const val ON_RETRY: String = "on_retry"
    public const val PRE_BATCH_ITEM: String = "pre_batch_item"
    public const val POST_BATCH_ITEM: String = "post_batch_item"
}

public object HookErrorStrategy {
    public const val SKIP: String = "skip"
    public const val ABORT: String = "abort"
}

/** Guard / blocklist PII / max-length: `input`, `output`, or `both`. */
public object GuardSide {
    public const val INPUT: String = "input"
    public const val OUTPUT: String = "output"
    public const val BOTH: String = "both"
}

public object BlocklistAction {
    public const val BLOCK: String = "block"
    public const val REDACT: String = "redact"
}

public object MaxLengthStrategy {
    public const val BLOCK: String = "block"
    public const val TRUNCATE: String = "truncate"
}

public object BatchErrorStrategy {
    public const val CONTINUE: String = "continue"
    public const val SKIP: String = "skip"
    public const val FAIL_FAST: String = "fail_fast"
}

/**
 * @param apiKey OpenAI-compatible API key.
 * @param model e.g. `gpt-5.4-nano-2026-03-17-mini`
 */
@Serializable
public data class ClientConfig(
    @SerialName("apiKey")
    public val apiKey: String,
    @SerialName("model")
    public val model: String = "gpt-5.4-nano-2026-03-17-mini",
    @SerialName("baseUrl")
    public val baseUrl: String = "https://api.openai.com",
    @SerialName("systemPrompt")
    public val systemPrompt: String? = null,
    @SerialName("maxToolRounds")
    public val maxToolRounds: Int = 16,
    @SerialName("maxRetries")
    public val maxRetries: Int = 3,
    @SerialName("retryInitialDelayMs")
    public val retryInitialDelayMs: Int = 50,
    @SerialName("retryMaxDelayMs")
    public val retryMaxDelayMs: Int = 2000,
    @SerialName("retryMultiplier")
    public val retryMultiplier: Double = 2.0,
    @SerialName("requestsPerSecond")
    public val requestsPerSecond: Int? = null,
    @SerialName("timeoutSecs")
    public val timeoutSecs: Int = 60,
    @SerialName("connectTimeoutSecs")
    public val connectTimeoutSecs: Int = 30,
    @SerialName("maxOutputRetries")
    public val maxOutputRetries: Int = 3,
    @SerialName("poolMaxIdlePerHost")
    public val poolMaxIdlePerHost: Int = 50,
    @SerialName("poolIdleTimeoutSecs")
    public val poolIdleTimeoutSecs: Int? = null,
    @SerialName("reasoningEffort")
    public val reasoningEffort: String? = null,
    @SerialName("modelFallbackModels")
    public val modelFallbackModels: List<String>? = null,
    @SerialName("apiKeys")
    public val apiKeys: Map<String, String>? = null,
    @SerialName("requestsPerSecondFor")
    public val requestsPerSecondFor: Map<String, Int>? = null,
    @SerialName("maxUploadBytes")
    public val maxUploadBytes: Int? = null,
) {
    public fun toConfigJsonString(): String = configJson.encodeToString(this)
}

@Serializable
public data class CompletionOutcome(
    public val content: String? = null,
    @SerialName("rounds")
    public val rounds: Int,
    @SerialName("usage")
    public val usage: JsonElement? = null,
    @SerialName("requestId")
    public val requestId: String,
    @SerialName("modelUsed")
    public val modelUsed: String? = null,
    @SerialName("messages")
    public val messages: JsonElement? = null,
)

@Serializable
public data class ResponseOutcome(
    public val id: String,
    public val content: String? = null,
    @SerialName("rounds")
    public val rounds: Int,
    @SerialName("usage")
    public val usage: JsonElement? = null,
    @SerialName("requestId")
    public val requestId: String,
    @SerialName("modelUsed")
    public val modelUsed: String,
)

@Serializable
public data class ResponseStreamOutcome(
    public val id: String,
    public val content: String,
    @SerialName("usage")
    public val usage: JsonElement? = null,
    @SerialName("requestId")
    public val requestId: String,
    @SerialName("modelUsed")
    public val modelUsed: String,
)

@Serializable
public data class StreamOutcome(
    public val content: String,
    @SerialName("finishReason")
    public val finishReason: String? = null,
    @SerialName("usage")
    public val usage: JsonElement? = null,
    @SerialName("requestId")
    public val requestId: String,
)

@Serializable
public data class UploadedFile(
    public val provider: String,
    @SerialName("fileId")
    public val fileId: String,
    public val filename: String,
    public val bytes: Int,
    public val purpose: String,
)

@Serializable
public data class OpenAiChatMessage(
    public val role: String,
    public val content: String? = null,
)

/**
 * Agent persona spec for [SuperglueClient.runAgent] (OpenAI/HTTP + tools; matches native [AgentSpecJson]).
 */
@Serializable
public data class AgentRunSpec(
    public val name: String,
    public val persona: String,
    public val goals: List<String>? = null,
    public val constraints: List<String>? = null,
    @SerialName("model")
    public val model: String = "",
    @SerialName("maxToolRounds")
    public val maxToolRounds: Int = 16,
    @SerialName("maxOutputRetries")
    public val maxOutputRetries: Int = 3,
    @SerialName("systemPrompt")
    public val systemPrompt: String? = null,
    @SerialName("reasoningEffort")
    public val reasoningEffort: String? = null,
) {
    public fun toSpecJsonString(): String = configJson.encodeToString(this)
}

@Serializable
public data class EngineHttpConfig(
    @SerialName("apiKey")
    public val apiKey: String,
    @SerialName("baseUrl")
    public val baseUrl: String? = null,
    @SerialName("model")
    public val model: String? = null,
    @SerialName("maxRetries")
    public val maxRetries: Int = 3,
    @SerialName("retryInitialDelayMs")
    public val retryInitialDelayMs: Int = 50,
    @SerialName("retryMaxDelayMs")
    public val retryMaxDelayMs: Int = 2000,
    @SerialName("retryMultiplier")
    public val retryMultiplier: Double = 2.0,
    @SerialName("requestsPerSecond")
    public val requestsPerSecond: Int? = null,
    @SerialName("timeoutSecs")
    public val timeoutSecs: Int = 60,
    @SerialName("connectTimeoutSecs")
    public val connectTimeoutSecs: Int = 30,
    @SerialName("poolMaxIdlePerHost")
    public val poolMaxIdlePerHost: Int = 50,
    @SerialName("poolIdleTimeoutSecs")
    public val poolIdleTimeoutSecs: Int? = null,
) {
    public fun toConfigJsonString(): String = configJson.encodeToString(this)
}

@Serializable
public data class BatchRequestItem(
    public val id: String? = null,
    @SerialName("systemPrompt")
    public val systemPrompt: String? = null,
    public val prompt: String,
)

@Serializable
public data class BatchItemOutcome(
    public val id: String? = null,
    public val success: Boolean,
    public val content: String? = null,
    public val error: String? = null,
    @SerialName("rounds")
    public val rounds: Int? = null,
    public val usage: JsonElement? = null,
    @SerialName("elapsedSecs")
    public val elapsedSecs: Double? = null,
)

@Serializable
public data class BatchOutcome(
    public val results: List<BatchItemOutcome>,
    @SerialName("totalRequests")
    public val totalRequests: Int,
    public val successful: Int,
    public val failed: Int,
    @SerialName("elapsedSecs")
    public val elapsedSecs: Double? = null,
    @SerialName("totalUsage")
    public val totalUsage: JsonElement? = null,
)

private val jsonParser = Json { ignoreUnknownKeys = true }

/**
 * In-process superglue client (mirrors JS/Python bindings).
 */
public class SuperglueClient
    @PublishedApi
    internal constructor(
        public val handle: Long,
    ) : AutoCloseable {
        public companion object {
            public fun open(
                config: ClientConfig,
                statusEmitter: StatusEmitter? = null,
            ): SuperglueClient {
                val h =
                    if (statusEmitter != null) {
                        SuperglueNativeJni.clientCreateWithEmitter(
                            config.toConfigJsonString(),
                            statusEmitter.handle,
                        )
                    } else {
                        SuperglueNativeJni.clientCreate(config.toConfigJsonString())
                    }
                if (h == 0L) throw SuperglueException("clientCreate failed")
                return SuperglueClient(h)
            }
        }

        public fun complete(
            userMessage: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
            reasoningEffort: String? = null,
        ): CompletionOutcome {
            val s =
                SuperglueNativeJni.clientComplete(
                    handle,
                    userMessage,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                    reasoningEffort.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(CompletionOutcome.serializer(), s)
        }

        public fun completeMessages(
            messages: List<OpenAiChatMessage>,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome = completeMessagesJson(
            configJson.encodeToString(
                ListSerializer(OpenAiChatMessage.serializer()),
                messages,
            ),
            timeoutSecs,
            connectTimeoutSecs,
            requestId,
        )

        public fun completeMessagesJson(
            messagesJson: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome {
            val s =
                SuperglueNativeJni.clientCompleteMessages(
                    handle,
                    messagesJson,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(CompletionOutcome.serializer(), s)
        }

        public fun stream(
            userMessage: String,
            onToken: StreamTokenCallback,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): StreamOutcome {
            val s =
                SuperglueNativeJni.clientStream(
                    handle,
                    userMessage,
                    onToken,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(StreamOutcome.serializer(), s)
        }

        public fun uploadFile(
            path: String,
            purpose: String = "user_data",
            provider: String? = null,
        ): UploadedFile {
            val s =
                SuperglueNativeJni.clientUploadFile(
                    handle,
                    path,
                    purpose,
                    provider.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(UploadedFile.serializer(), s)
        }

        public fun messageWithFileBytes(
            filename: String,
            bytes: ByteArray,
            text: String? = null,
        ): String {
            return SuperglueNativeJni.clientMessageWithFileBytes(
                filename,
                bytes,
                text.orEmpty(),
            ) ?: throw SuperglueException("native returned null")
        }

        public fun completeResponse(
            userMessage: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
            reasoningEffort: String? = null,
        ): ResponseOutcome {
            val s =
                SuperglueNativeJni.clientCompleteResponse(
                    handle,
                    userMessage,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                    reasoningEffort.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(ResponseOutcome.serializer(), s)
        }

        public fun streamResponse(
            userMessage: String,
            onToken: StreamTokenCallback,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): ResponseStreamOutcome {
            val s =
                SuperglueNativeJni.clientStreamResponse(
                    handle,
                    userMessage,
                    onToken,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(ResponseStreamOutcome.serializer(), s)
        }

        public fun connectMcpStdio(
            command: String,
            argsJson: String = "[]",
            envJson: String = "",
            prefix: String? = null,
        ) {
            SuperglueNativeJni.clientConnectMcpStdio(
                handle,
                command,
                argsJson,
                envJson,
                prefix.orEmpty(),
            )
        }

        public fun connectMcpHttp(url: String, prefix: String? = null) {
            SuperglueNativeJni.clientConnectMcpHttp(handle, url, prefix.orEmpty())
        }

        public fun batch(
            requests: List<BatchRequestItem>,
            maxConcurrent: Int = 5,
            errorStrategy: String = BatchErrorStrategy.CONTINUE,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
        ): BatchOutcome {
            val requestsJson =
                configJson.encodeToString(
                    ListSerializer(BatchRequestItem.serializer()),
                    requests,
                )
            val s =
                SuperglueNativeJni.clientBatch(
                    handle,
                    requestsJson,
                    maxConcurrent,
                    errorStrategy,
                    timeoutSecs,
                    connectTimeoutSecs,
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(BatchOutcome.serializer(), s)
        }

        public fun runAgent(
            spec: AgentRunSpec,
            userMessage: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome = runAgentJson(spec.toSpecJsonString(), userMessage, timeoutSecs, connectTimeoutSecs, requestId)

        public fun runAgentJson(
            specJson: String,
            userMessage: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome {
            val s =
                SuperglueNativeJni.clientRunAgent(
                    handle,
                    specJson,
                    userMessage,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(CompletionOutcome.serializer(), s)
        }

        public fun registerTool(
            name: String,
            description: String,
            parametersJson: String,
            callback: JsonCallback,
        ) {
            SuperglueNativeJni.clientRegisterTool(handle, name, description, parametersJson, callback)
        }

        public fun cancel() {
            if (handle != 0L) {
                SuperglueNativeJni.clientCancel(handle)
            }
        }

        public fun registerHook(
            stage: String,
            name: String,
            errorStrategy: String,
            callback: JsonCallback,
        ) {
            SuperglueNativeJni.clientRegisterHook(handle, stage, name, errorStrategy, callback)
        }

        public fun toolsRegistryPtr(): Long =
            SuperglueNativeJni.clientToolsRegistryPtr(handle)

        public fun hooksRegistryPtr(): Long =
            SuperglueNativeJni.clientHooksRegistryPtr(handle)

        public fun registerGuardrail(
            callback: GuardrailCallback,
            stage: String = GuardSide.BOTH,
            name: String = "custom_guard",
        ) {
            SuperglueNativeJni.clientRegisterGuardrail(handle, callback, stage, name)
        }

        public fun addBlocklistGuardrail(
            patterns: List<String>,
            action: String = BlocklistAction.BLOCK,
            stage: String = GuardSide.BOTH,
            name: String = "blocklist",
        ) {
            val patternsJson =
                configJson.encodeToString(
                    ListSerializer(String.serializer()),
                    patterns,
                )
            SuperglueNativeJni.clientAddBlocklistGuardrail(handle, patternsJson, action, stage, name)
        }

        public fun addMaxLengthGuardrail(
            maxInput: Int = -1,
            maxOutput: Int = -1,
            strategy: String = MaxLengthStrategy.BLOCK,
            name: String = "max_length",
        ) {
            SuperglueNativeJni.clientAddMaxLengthGuardrail(handle, maxInput, maxOutput, strategy, name)
        }

        public fun addPiiRedactGuardrail(
            stage: String = GuardSide.BOTH,
            name: String = "pii_redact",
        ) {
            SuperglueNativeJni.clientAddPiiGuardrail(handle, stage, name)
        }

        public fun toConversation(): SuperglueConversation = SuperglueConversation.open(this)

        override fun close() {
            if (handle != 0L) {
                SuperglueNativeJni.clientDestroy(handle)
            }
        }
    }

/**
 * Conversation view over a [SuperglueClient] (pushes turns then completes with tools + hooks).
 */
public class SuperglueConversation
    @PublishedApi
    internal constructor(
        private val handle: Long,
    ) : AutoCloseable {
        public companion object {
            public fun open(client: SuperglueClient): SuperglueConversation {
                val h = SuperglueNativeJni.conversationFromClient(client.handle)
                if (h == 0L) throw SuperglueException("conversationFromClient failed")
                return SuperglueConversation(h)
            }
        }

        public fun pushUser(text: String) {
            SuperglueNativeJni.conversationPushUser(handle, text)
        }

        public fun pushAssistant(text: String) {
            SuperglueNativeJni.conversationPushAssistant(handle, text)
        }

        public fun getMessagesJson(): String = SuperglueNativeJni.conversationGetMessagesJson(handle)
            ?: "{}"

        public fun complete(
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome {
            val s =
                SuperglueNativeJni.conversationComplete(
                    handle,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(CompletionOutcome.serializer(), s)
        }

        override fun close() {
            if (handle != 0L) {
                SuperglueNativeJni.conversationDestroy(handle)
            }
        }
    }

/**
 * Standalone agent engine (shared tools with HTTP config).
 */
public class SuperglueAgentEngine
    @PublishedApi
    internal constructor(
        public val handle: Long,
    ) : AutoCloseable {
        public companion object {
            public fun open(
                spec: AgentRunSpec,
                http: EngineHttpConfig,
            ): SuperglueAgentEngine = openJson(spec.toSpecJsonString(), http.toConfigJsonString())

            public fun openJson(
                specJson: String,
                httpConfigJson: String,
            ): SuperglueAgentEngine {
                val h = SuperglueNativeJni.agentEngineCreate(specJson, httpConfigJson)
                if (h == 0L) throw SuperglueException("agentEngineCreate failed")
                return SuperglueAgentEngine(h)
            }
        }

        public fun registerTool(
            name: String,
            description: String,
            parametersJson: String,
            callback: JsonCallback,
        ) {
            SuperglueNativeJni.agentEngineRegisterTool(handle, name, description, parametersJson, callback)
        }

        public fun registerHook(
            stage: String,
            name: String,
            errorStrategy: String,
            callback: JsonCallback,
        ) {
            SuperglueNativeJni.agentEngineRegisterHook(handle, stage, name, errorStrategy, callback)
        }

        public fun registerGuardrail(
            callback: GuardrailCallback,
            stage: String = GuardSide.BOTH,
            name: String = "custom_guard",
        ) {
            SuperglueNativeJni.agentEngineRegisterGuardrail(handle, callback, stage, name)
        }

        public fun run(
            userMessage: String,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): CompletionOutcome {
            val s =
                SuperglueNativeJni.agentEngineRun(
                    handle,
                    userMessage,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(CompletionOutcome.serializer(), s)
        }

        public fun stream(
            userMessage: String,
            onToken: StreamTokenCallback,
            timeoutSecs: Long = -1L,
            connectTimeoutSecs: Long = -1L,
            requestId: String? = null,
        ): StreamOutcome {
            val s =
                SuperglueNativeJni.agentEngineStream(
                    handle,
                    userMessage,
                    onToken,
                    timeoutSecs,
                    connectTimeoutSecs,
                    requestId.orEmpty(),
                ) ?: throw SuperglueException("native returned null")
            return jsonParser.decodeFromString(StreamOutcome.serializer(), s)
        }

        override fun close() {
            if (handle != 0L) {
                SuperglueNativeJni.agentEngineDestroy(handle)
            }
        }
    }

/**
 * Fan-out dispatcher for typed LLM process events (gluellm #351 parity).
 */
public class StatusEmitter : AutoCloseable {
    internal val handle: Long = SuperglueNativeJni.statusEmitterCreate()

    public fun subscribe(callback: ProcessEventCallback) {
        SuperglueNativeJni.statusEmitterSubscribe(handle, callback)
    }

    override fun close() {
        if (handle != 0L) {
            SuperglueNativeJni.statusEmitterDestroy(handle)
        }
    }
}
