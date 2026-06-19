package com.superglue.kt;

@FunctionalInterface
public interface ProcessEventCallback {
    void onEvent(String eventJson);
}
