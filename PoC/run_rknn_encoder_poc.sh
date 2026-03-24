#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RKNN_POC_HOST="${RKNN_POC_HOST:-127.0.0.1}"
export RKNN_POC_PORT="${RKNN_POC_PORT:-7861}"
export RKNN_POC_RKNN_PYTHON="${RKNN_POC_RKNN_PYTHON:-/home/pi/npu_env/bin/python}"

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/PoC/rknn_encoder_app.py"
