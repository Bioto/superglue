# Superglue

Superglue is a Rust library and CLI for LLM orchestration, with Python, JavaScript, and Kotlin bindings.

## Packages

The Rust core and all language bindings share one version.

| Package | Path |
|---------|------|
| Core | `.` (`superglue` crate) |
| Python | `superglue-py/` |
| JavaScript (napi-rs) | `superglue-js/` |
| Kotlin (JNI) | `superglue-kt/` |

See [PACKAGES.md](PACKAGES.md) and [MONOREPO.md](MONOREPO.md) for more detail.

## Quick start

Clone the repo, then build and test from the repo root:

```bash
cargo build
cargo test
```

Optional MCP tests:

```bash
cargo test --features mcp
```

**Version source of truth:** `[workspace.package].version` in [`Cargo.toml`](Cargo.toml).

After you change that field, run:

```bash
./scripts/sync-superglue-versions.sh
./scripts/check-superglue-version-parity.sh
```

## Design notes

Maintainer design notes live in [`_docs/README.md`](_docs/README.md).

Multi-provider notes live in [`docs/multi-provider.md`](docs/multi-provider.md).

## License

BSD-4-Clause. See [LICENSE](LICENSE).

Public materials that mention this software must display the acknowledgement in the LICENSE file.

This license is GPL-incompatible.
