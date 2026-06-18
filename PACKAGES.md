# Superglue packages

The Rust core (repo root) and all language bindings are developed **together** and share a single version.

| Package | Path |
|---------|------|
| Core | `.` (`superglue` crate) |
| Python | `superglue-py/` |
| JavaScript (napi-rs) | `superglue-js/` |
| Kotlin (JNI) | `superglue-kt/` |

## Version

**Source of truth:** `[workspace.package].version` in [`Cargo.toml`](Cargo.toml).

When bumping the release version, edit that field only, then run:

```bash
./scripts/sync-superglue-versions.sh
./scripts/check-superglue-version-parity.sh
```

Bindings always depend on the in-repo `superglue` crate with the same feature set (`mcp` enabled) so API surface stays aligned with the core.

## Build (Rust workspace)

```bash
cargo build
cargo test --features mcp
```
