#!/usr/bin/env bash
# Keep superglue core and all binding package versions in sync.
# Source of truth: Cargo.toml [workspace.package].version at repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_CARGO="$ROOT/Cargo.toml"

if [[ ! -f "$WORKSPACE_CARGO" ]]; then
  echo "Missing $WORKSPACE_CARGO"
  exit 1
fi

VERSION="$(grep -E '^version = ' "$WORKSPACE_CARGO" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "$VERSION" ]]; then
  echo "Could not read workspace version from $WORKSPACE_CARGO"
  exit 1
fi

echo "Syncing superglue version: $VERSION"

echo "$VERSION" > "$ROOT/VERSION"

JS_PKG="$ROOT/superglue-js/package.json"
if [[ -f "$JS_PKG" ]]; then
  if command -v node >/dev/null 2>&1; then
    node -e "
      const fs = require('fs');
      const p = process.argv[1];
      const v = process.argv[2];
      const j = JSON.parse(fs.readFileSync(p, 'utf8'));
      j.version = v;
      fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n');
    " "$JS_PKG" "$VERSION"
    echo "  updated $JS_PKG"
  else
    echo "  skip $JS_PKG (node not found)"
  fi
fi

KT_GRADLE="$ROOT/superglue-kt/superglue-android/build.gradle.kts"
if [[ -f "$KT_GRADLE" ]]; then
  if grep -q 'superglueLibraryVersion' "$KT_GRADLE"; then
    sed -i "s/superglueLibraryVersion = \"[^\"]*\"/superglueLibraryVersion = \"$VERSION\"/" "$KT_GRADLE"
    echo "  updated superglueLibraryVersion in build.gradle.kts"
  fi
fi

echo "Done."
