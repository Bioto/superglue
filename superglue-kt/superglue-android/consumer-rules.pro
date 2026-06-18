# Keep native JNI and Kotlin entrypoints for R8
-keep class com.superglue.kt.SuperglueNativeJni { *; }
-keep class com.superglue.kt.JsonCallback { *; }
-keep class com.superglue.kt.GuardrailCallback { *; }
-keep class com.superglue.kt.StreamTokenCallback { *; }
