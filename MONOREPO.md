# Superglue monorepo layout

| Component | Path | Role |
|-----------|------|------|
| **superglue** (core) | repo root | Rust library + CLI |
| **superglue-js** | `superglue-js/` | Node.js napi-rs bindings |
| **superglue-py** | `superglue-py/` | Python PyO3 bindings |
| **superglue-kt** | `superglue-kt/` | Kotlin/Android JNI bindings |

Package overview and version sync: [PACKAGES.md](PACKAGES.md).

In **thestack**, this repo is checked out as submodule `projects/superglue`. Loom and other consumers use `path = "../superglue"` from sibling submodules.
