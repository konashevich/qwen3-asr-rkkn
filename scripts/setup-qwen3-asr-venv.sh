#!/usr/bin/env bash
set -euo pipefail

venv_dir="${1:-.venv-qwen3-asr}"

python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements-qwen3-asr.txt

cat <<EOF
Qwen3-ASR venv created at: $venv_dir
Activate it with:
  source "$venv_dir/bin/activate"

Optional CUDA/vLLM path:
  python -m pip install -U "qwen-asr[vllm]"
EOF