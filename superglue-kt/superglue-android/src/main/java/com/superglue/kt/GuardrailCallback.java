package com.superglue.kt;

@FunctionalInterface
public interface GuardrailCallback {
    String invoke(String stage, String content);
}
