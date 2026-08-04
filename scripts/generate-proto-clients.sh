#!/usr/bin/env bash
# Generate gRPC client stubs for Python and JavaScript from proto/superglue.proto.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO="$ROOT/proto/superglue.proto"
PY_OUT="$ROOT/superglue-py/superglue/grpc"
JS_OUT="$ROOT/superglue-js/generated"

mkdir -p "$PY_OUT" "$JS_OUT"

PYTHON="python3"
if [[ -x "$ROOT/.venv-proto/bin/python" ]]; then
  PYTHON="$ROOT/.venv-proto/bin/python"
fi

if command -v "$PYTHON" >/dev/null 2>&1; then
  "$PYTHON" -m grpc_tools.protoc \
    -I "$ROOT/proto" \
    --python_out="$PY_OUT" \
    --grpc_python_out="$PY_OUT" \
    --pyi_out="$PY_OUT" \
    "$PROTO"
  echo "Generated Python stubs in $PY_OUT"
else
  echo "python3 not found; skipping Python stub generation" >&2
  exit 1
fi

if command -v npx >/dev/null 2>&1; then
  npx --yes grpc_tools_node_protoc_ts \
    --proto_path="$ROOT/proto" \
    --grpcLib=@grpc/grpc-js \
    --outDir="$JS_OUT" \
    superglue.proto 2>/dev/null || echo "JS stub generation skipped (checked-in client.ts loader is primary)" >&2
else
  echo "npx not found; skipping JS stub generation" >&2
fi

echo "Proto client generation complete. Rust tonic client is built via --features grpc."
