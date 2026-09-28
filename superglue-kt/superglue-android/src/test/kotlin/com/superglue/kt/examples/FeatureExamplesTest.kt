package com.superglue.kt.examples

import com.superglue.kt.AgentRunSpec
import com.superglue.kt.ClientConfig
import com.superglue.kt.JsonCallback
import com.superglue.kt.ProcessEventCallback
import com.superglue.kt.StatusEmitter
import com.superglue.kt.StreamTokenCallback
import com.superglue.kt.SuperglueClient
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Runnable examples mirroring `superglue-py/examples/22–31`.
 * Set `OPENAI_API_KEY` in the environment to run live sections.
 */
class FeatureExamplesTest {
    private val apiKey = System.getenv("OPENAI_API_KEY").orEmpty()
    private val model = System.getenv("OPENAI_MODEL") ?: "gpt-5.4-nano-2026-03-17-mini"

    @Test
    fun example22_status_events() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 22 live.")
            return
        }
        val emitter = StatusEmitter()
        emitter.subscribe(
            ProcessEventCallback { json ->
                val obj = Json.parseToJsonElement(json).jsonObject
                println("[${obj["kind"]!!.jsonPrimitive.content}] model=${obj["model"]!!.jsonPrimitive.content}")
            },
        )
        val client =
            SuperglueClient.open(
                ClientConfig(apiKey = apiKey, model = model),
                statusEmitter = emitter,
            )
        val result = client.complete("Say hello in one word.")
        assertTrue(result.content?.isNotEmpty() == true)
        emitter.close()
        client.close()
    }

    @Test
    fun example23_reasoning_effort() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 23 live.")
            return
        }
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = model,
                    reasoningEffort = "medium",
                    systemPrompt = "Answer in one short sentence.",
                ),
            )
        val medium = client.complete("What is 17 + 25?")
        assertTrue(medium.content?.isNotEmpty() == true)

        val high =
            client.complete(
                "Name one benefit of higher reasoning effort.",
                reasoningEffort = "high",
            )
        assertTrue(high.content?.isNotEmpty() == true)

        val spec =
            AgentRunSpec(
                name = "DeepThinker",
                persona = "a careful analyst",
                goals = listOf("Give precise answers"),
                reasoningEffort = "high",
            )
        val agentResult =
            client.runAgent(spec, "Why might exponential backoff beat fixed-interval retry?")
        assertTrue(agentResult.content?.isNotEmpty() == true)
        client.close()
    }

    @Test
    fun example24_connection_pool_defaults() {
        val cfg = ClientConfig(apiKey = "k")
        assertEquals(30, cfg.connectTimeoutSecs)
        assertEquals(50, cfg.poolMaxIdlePerHost)
        assertEquals(60, cfg.timeoutSecs)

        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 24 live section.")
            return
        }
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = model,
                    connectTimeoutSecs = 30,
                    poolMaxIdlePerHost = 50,
                    timeoutSecs = 90,
                ),
            )
        val result = client.complete("Say 'pool ok' in two words.")
        assertTrue(result.content?.isNotEmpty() == true)
        client.close()
    }

    @Test
    fun example25_model_fallback() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 25 live.")
            return
        }
        val backup = System.getenv("OPENAI_FALLBACK_MODEL") ?: model
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = model,
                    modelFallbackModels = listOf(model, backup),
                ),
            )
        val result = client.complete("Reply with exactly: fallback ok")
        assertTrue(result.content?.isNotEmpty() == true)
        client.close()
    }

    @Test
    fun example26_stream_response() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 26 live.")
            return
        }
        val client = SuperglueClient.open(ClientConfig(apiKey = apiKey, model = model))
        val tokens = mutableListOf<String>()
        val outcome =
            client.streamResponse(
                "Count from 1 to 3 separated by spaces.",
                StreamTokenCallback { tokens.add(it) },
            )
        assertTrue(outcome.content.isNotEmpty())
        assertTrue(tokens.isNotEmpty())
        client.close()
    }

    @Test
    fun example27_complete_response() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 27 live.")
            return
        }
        val client = SuperglueClient.open(ClientConfig(apiKey = apiKey, model = model))
        client.registerTool(
            "echo",
            "Echo JSON",
            "{\"type\":\"object\"}",
            JsonCallback { args -> "{\"echo\":$args}" },
        )
        val result =
            client.completeResponse(
                "Call echo with {\"n\":1} and reply in one short sentence.",
            )
        assertTrue(result.content?.isNotEmpty() == true)
        assertTrue(result.rounds >= 1)
        client.close()
    }

    @Test
    fun example29_multi_provider() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 29 live.")
            return
        }
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY").orEmpty()
        val apiKeys = mutableMapOf("openai" to apiKey)
        if (anthropicKey.isNotEmpty()) {
            apiKeys["anthropic"] = anthropicKey
        }
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = "openai:gpt-4o-mini",
                    apiKeys = apiKeys,
                ),
            )
        val out = client.complete("Say hello in one word.")
        assertTrue(out.content?.isNotEmpty() == true)
        client.close()
    }

    @Test
    fun example30_streaming_tools() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 30 live.")
            return
        }
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = model,
                ),
            )
        client.registerTool(
            "echo",
            "Echo JSON",
            "{\"type\":\"object\"}",
            JsonCallback { args -> "{\"echo\":$args}" },
        )
        val tokens = mutableListOf<String>()
        val out =
            client.stream(
                "Call echo with {\"x\":1} then say done",
                StreamTokenCallback { tokens.add(it) },
            )
        assertTrue(out.content.isNotEmpty())
        client.close()
    }

    @Test
    fun example31_file_upload_inline() {
        if (apiKey.isEmpty()) {
            println("Set OPENAI_API_KEY to run example 31 live.")
            return
        }
        val client =
            SuperglueClient.open(
                ClientConfig(
                    apiKey = apiKey,
                    model = model,
                ),
            )
        // OpenAI inline chat file parts require application/pdf data URLs.
        val pdf =
            """
%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 200 200]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 24 Tf 20 100 Td (Hi) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000060 00000 n 
0000000114 00000 n 
0000000220 00000 n 
0000000314 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
380
%%EOF
            """.trimIndent()
        val bytes = pdf.encodeToByteArray()
        val msgJson =
            client.messageWithFileBytes(
                "sample.pdf",
                bytes,
                "Summarize in one short sentence.",
            )
        val out = client.completeMessagesJson("[$msgJson]")
        assertTrue(out.content?.isNotEmpty() == true)
        client.close()
    }
}
