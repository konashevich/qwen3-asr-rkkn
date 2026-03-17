#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/dist/rk3588_encoder_runtime"

mkdir -p "$BUNDLE_DIR"

cp "$ROOT_DIR/validate_converted.py" "$BUNDLE_DIR/"
cp "$ROOT_DIR/scripts/setup-rk3588-runtime.sh" "$BUNDLE_DIR/"
cp "$ROOT_DIR/outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn" "$BUNDLE_DIR/"
cp "$ROOT_DIR/outputs/runtime_reference/input_features.npy" "$BUNDLE_DIR/"
cp "$ROOT_DIR/outputs/runtime_reference/encoder_reference_output.npy" "$BUNDLE_DIR/"
cp "$ROOT_DIR/outputs/runtime_reference/runtime_reference.json" "$BUNDLE_DIR/"

cat > "$BUNDLE_DIR/RUN_ON_RK3588.txt" <<'EOF'
1. Copy this folder to the RK3588 board.
2. Put the correct Rockchip rknn-toolkit-lite2 wheel for your board OS in this folder.
3. Create the runtime environment:
  bash setup-rk3588-runtime.sh ./rknn_toolkit_lite2-*.whl
4. Run:
  ./.venv-rk3588-runtime/bin/python validate_converted.py \
     --model qwen3_asr_encoder_single_chunk_rk3588.rknn \
     --input-features input_features.npy \
     --reference-output encoder_reference_output.npy
5. Inspect validation_report.json and device_output.npy.
EOF

echo "Created runtime bundle at $BUNDLE_DIR"