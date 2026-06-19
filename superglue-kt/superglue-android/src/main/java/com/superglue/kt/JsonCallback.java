package com.superglue.kt;

@FunctionalInterface
public interface JsonCallback {
    String invoke(String argsJson);
}
