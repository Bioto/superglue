package com.superglue.kt.examples

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Documents which numbered superglue-py/js examples have Kotlin JUnit coverage.
 * Live sections require OPENAI_API_KEY (see [FeatureExamplesTest]).
 */
class ExamplesRunnerTest {
    private val coveredExamples =
        listOf(22, 23, 24, 25, 26, 27, 29, 30, 31)

    @Test
    fun documentsCoveredExampleNumbers() {
        assertTrue(coveredExamples.contains(25))
        assertTrue(coveredExamples.size >= 9)
    }
}
