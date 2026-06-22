package com.superglue.kt;

/**
 * Low-level JNI surface for {@code libsuperglue_kt.so}. Prefer {@link Superglue} / {@link SuperglueClient}.
 */
public final class SuperglueNativeJni {
    static {
        System.loadLibrary("superglue_kt");
    }

    private SuperglueNativeJni() {}

    public static native String version();

    public static native long clientCreate(String configJson);

    public static native long clientCreateWithEmitter(String configJson, long emitterHandle);

    public static native void clientDestroy(long handle);

    public static native void clientCancel(long handle);

    public static native void clientRegisterTool(
        long handle,
        String name,
        String description,
        String parametersJson,
        JsonCallback callback
    );

    public static native void clientRegisterHook(
        long handle,
        String stage,
        String name,
        String errorStrategy,
        JsonCallback callback
    );

    public static native void clientRegisterGuardrail(
        long handle,
        GuardrailCallback callback,
        String stage,
        String name
    );

    public static native void clientAddBlocklistGuardrail(
        long handle,
        String patternsJson,
        String action,
        String stage,
        String name
    );

    public static native void clientAddMaxLengthGuardrail(
        long handle,
        int maxInput,
        int maxOutput,
        String strategy,
        String name
    );

    public static native void clientAddPiiGuardrail(long handle, String stage, String name);

    public static native String clientRunAgent(
        long handle,
        String specJson,
        String userMessage,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native String clientComplete(
        long handle,
        String userMessage,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId,
        String reasoningEffort
    );

    public static native long statusEmitterCreate();

    public static native void statusEmitterDestroy(long handle);

    public static native void statusEmitterSubscribe(long handle, ProcessEventCallback callback);

    public static native String clientCompleteMessages(
        long handle,
        String messagesJson,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native String clientStream(
        long handle,
        String userMessage,
        StreamTokenCallback onToken,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native String clientUploadFile(
        long handle,
        String path,
        String purpose,
        String provider
    );

    public static native String clientMessageWithFileBytes(
        String filename,
        byte[] fileBytes,
        String text
    );

    public static native String clientCompleteResponse(
        long handle,
        String userMessage,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId,
        String reasoningEffort
    );

    public static native String clientStreamResponse(
        long handle,
        String userMessage,
        StreamTokenCallback onToken,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native void clientConnectMcpStdio(
        long handle,
        String command,
        String argsJson,
        String envJson,
        String prefix
    );

    public static native void clientConnectMcpHttp(long handle, String url, String prefix);

    public static native String clientBatch(
        long handle,
        String requestsJson,
        int maxConcurrent,
        String errorStrategy,
        long timeoutSecs,
        long connectTimeoutSecs
    );

    // --- conversation ---

    public static native long conversationFromClient(long clientHandle);

    public static native void conversationDestroy(long handle);

    public static native void conversationPushUser(long handle, String text);

    public static native void conversationPushAssistant(long handle, String text);

    public static native String conversationGetMessagesJson(long handle);

    public static native String conversationComplete(
        long handle,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    // --- agent engine ---

    public static native long agentEngineCreate(String specJson, String httpConfigJson);

    public static native void agentEngineDestroy(long handle);

    public static native void agentEngineRegisterTool(
        long handle,
        String name,
        String description,
        String parametersJson,
        JsonCallback callback
    );

    public static native void agentEngineRegisterHook(
        long handle,
        String stage,
        String name,
        String errorStrategy,
        JsonCallback callback
    );

    public static native void agentEngineRegisterGuardrail(
        long handle,
        GuardrailCallback callback,
        String stage,
        String name
    );

    public static native String agentEngineRun(
        long handle,
        String userMessage,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native String agentEngineStream(
        long handle,
        String userMessage,
        StreamTokenCallback onToken,
        long timeoutSecs,
        long connectTimeoutSecs,
        String requestId
    );

    public static native long clientToolsRegistryPtr(long handle);

    public static native long clientHooksRegistryPtr(long handle);
}
