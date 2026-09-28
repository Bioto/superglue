#!/usr/bin/env bash
# Build + install the extension into the active venv.
#
# Usage:  ./dev.sh
#
# Uses `maturin build` (not `maturin develop`) so uv installs a real wheel
# that it owns and will not reinstall/overwrite on the next `uv run`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Always build against the venv's own interpreter so the wheel ABI tag matches.
PYTHON="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv not found at $SCRIPT_DIR/.venv — run 'uv venv --python 3.14t' first."
  exit 1
fi

echo "==> building wheel (debug profile, interpreter: $PYTHON)..."
maturin build --profile dev --out dist --interpreter "$PYTHON"

WHEEL=$(ls -t dist/superglue*.whl | head -1)
echo "==> installing $WHEEL via uv..."
uv pip install --reinstall "$WHEEL"

echo "==> verifying..."
# Run from /tmp so the local ./superglue/ source directory does not shadow
# the installed package in site-packages.
cd /tmp && "$PYTHON" -c \
  "import superglue; assert hasattr(superglue.Client, 'stream'), 'stream missing!'; print('OK — stream() is available')"
cd "$SCRIPT_DIR"
