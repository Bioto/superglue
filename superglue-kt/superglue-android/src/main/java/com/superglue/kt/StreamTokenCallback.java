package com.superglue.kt;

@FunctionalInterface
public interface StreamTokenCallback {
    void onToken(String token);
}
