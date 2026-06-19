# superglue-kt (Kotlin / Android)

JNI bindings to the in-process [superglue](../superglue) Rust crate, with the same feature surface as [superglue-js](../superglue-js) and [superglue-py](../superglue-py).

## Layout

- `native/` – Rust cdylib `libsuperglue_kt.so` (`superglue_kt` crate) using the `jni` crate
- `superglue-android/` – Android library (Kotlin + Java `SuperglueNativeJni`)

## Build native (Android)

Prerequisites: Rust, Android NDK, [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk) (`cargo install cargo-ndk`).

From `superglue-kt/superglue-android`:

```bash
export ANDROID_NDK_HOME=/path/to/ndk   # e.g. $ANDROID_SDK_ROOT/ndk/28.x
./gradlew :superglue-android:buildRustJni
./gradlew :superglue-android:assembleDebug
```

If `ANDROID_NDK_HOME` (or `android.ndkPath` in `local.properties`) is set, `preBuild` runs `buildRustJni` automatically. Without NDK, configure the module in an environment that can compile the shared library, then re-open the project.

## Consuming from Butler

[Butler](/butler) includes `":superglue-android"` via [settings](../butler/settings.gradle.kts) and depends on the module from `:app`. Use `com.superglue.kt` (`version()`, `SuperglueClient`, etc.) and `com.butler.data.ai.SuperglueBridge` for a safe version probe when `.so` is absent on host tests.

## Examples

Numbered examples mirroring `superglue-js/examples` and `superglue-py/examples` are sketched in [examples/](examples/README.md).

## API contract

The JNI boundary follows `superglue-js/index.d.ts`: clients, tools, hooks, guardrails, streaming, batch, conversations, and agent engine.
