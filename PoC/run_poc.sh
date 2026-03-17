#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QWEN3_ASR_HF_HOME="${QWEN3_ASR_HF_HOME:-/mnt/merged_ssd/qwen3-asr-model-cache/huggingface}"
export HF_HOME="${HF_HOME:-$QWEN3_ASR_HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export QWEN3_ASR_HOST="${QWEN3_ASR_HOST:-127.0.0.1}"
export QWEN3_ASR_PORT="${QWEN3_ASR_PORT:-7861}"

mkdir -p "$HF_HOME"

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/PoC/app.py"