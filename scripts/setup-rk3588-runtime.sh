#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/rknn_toolkit_lite2-*.whl"
  exit 1
fi

WHEEL_PATH="$1"
RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$RUNTIME_DIR/.venv-rk3588-runtime"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install numpy "$WHEEL_PATH"
"$VENV_DIR/bin/python" - <<'PY'
from rknnlite.api import RKNNLite
print('RKNNLite import ok')
print('Available core constants:', [name for name in dir(RKNNLite) if name.startswith('NPU_CORE_')])
PY

echo
echo "Runtime environment ready in $VENV_DIR"
echo "Run the validator with:"
echo "  $VENV_DIR/bin/python validate_converted.py --model qwen3_asr_encoder_single_chunk_rk3588.rknn --input-features input_features.npy --reference-output encoder_reference_output.npy"