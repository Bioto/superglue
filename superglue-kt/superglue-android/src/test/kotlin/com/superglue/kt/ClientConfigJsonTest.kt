package com.superglue.kt

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

class ClientConfigJsonTest {
    @Test
    fun `client config serializes to keys expected by native`() {
        val cfg =
            ClientConfig(
                apiKey = "k",
                model = "m",
            )
        val s = cfg.toConfigJsonString()
        assertTrue(s.contains("apiKey"))
        assertTrue(s.contains("maxToolRounds"))
        val obj = Json.parseToJsonElement(s) as JsonObject
        assertEquals("k", obj["apiKey"]!!.jsonPrimitive.content)
    }
}
