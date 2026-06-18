#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_CARGO="$ROOT/Cargo.toml"

VERSION="$(grep -E '^version = ' "$WORKSPACE_CARGO" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "$VERSION" ]]; then
  echo "Could not read workspace version from $WORKSPACE_CARGO"
  exit 1
fi

check_file_version() {
  local label="$1"
  local file="$2"
  local expected="$3"
  if [[ ! -f "$file" ]]; then
    echo "SKIP $label ($file missing)"
    return 0
  fi
  if grep -qF "$expected" "$file"; then
    echo "OK   $label"
  else
    echo "FAIL $label — expected substring: $expected"
    exit 1
  fi
}

echo "Checking superglue version parity (workspace: $VERSION)"

check_file_version "superglue-js package.json" "$ROOT/superglue-js/package.json" "\"version\": \"${VERSION}\""
check_file_version "kt superglueLibraryVersion" "$ROOT/superglue-kt/superglue-android/build.gradle.kts" "superglueLibraryVersion = \"${VERSION}\""

if command -v cargo >/dev/null 2>&1; then
  for pkg in superglue superglue_js superglue_py superglue_kt; do
    got="$(cargo metadata --manifest-path "$WORKSPACE_CARGO" --format-version 1 --no-deps \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((p['version'] for p in d['packages'] if p['name']=='$pkg'), ''))")"
    if [[ "$got" == "$VERSION" ]]; then
      echo "OK   cargo $pkg"
    else
      echo "FAIL cargo $pkg — got '$got', want '$VERSION'"
      exit 1
    fi
  done
fi

echo "All version checks passed."
