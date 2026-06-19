// Root aggregator for superglue-kt. Module code lives in :superglue-android.

plugins {
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}

tasks.register("cleanRust") {
    doLast {
        val nativeDir = file("native")
        if (nativeDir.exists()) {
            project.exec {
                workingDir = nativeDir
                commandLine = listOf("cargo", "clean")
                isIgnoreExitValue = true
            }
        }
    }
}
