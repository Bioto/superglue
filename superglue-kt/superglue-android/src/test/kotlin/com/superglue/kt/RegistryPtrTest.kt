package com.superglue.kt

import com.superglue.kt.ClientConfig
import org.junit.Assert.assertTrue
import org.junit.Test

class RegistryPtrTest {
    @Test
    fun registryPtrsNonZero() {
        SuperglueClient.open(
            ClientConfig(apiKey = "sk-test", model = "gpt-5.4-nano-2026-03-17-mini"),
        ).use { client ->
            assertTrue(client.toolsRegistryPtr() > 0L)
            assertTrue(client.hooksRegistryPtr() > 0L)
        }
    }
}
