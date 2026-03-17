# Qwen3-ASR Board PoC

This folder contains a simple local transcription app for the FriendlyElec CM3588 board.

## What It Does

- Runs `Qwen/Qwen3-ASR-0.6B` locally on the board through the official `qwen-asr` Python runtime
- Exposes a small browser UI for file upload and browser microphone recording
- Uses SSD-backed Hugging Face cache by default so model weights do not fill the root filesystem
- Shows current board and NPU diagnostics on the homepage

## What It Does Not Do

- It does not run end-to-end Qwen3-ASR natively through RKNN today
- It does not use the current reduced encoder `.rknn` artifact for production transcription

The current board-side RKNN validator still fails with `invalid RKNN_MAGIC` / `RKNN_ERR_MODEL_INVALID`, so the operational path here is the verified Python runtime.

## Run

From the workspace root:

```bash
bash PoC/run_poc.sh
```

Open:

```text
http://127.0.0.1:7861
```

## Defaults

- Model: `Qwen/Qwen3-ASR-0.6B`
- Cache root: `/mnt/merged_ssd/qwen3-asr-model-cache/huggingface`
- Host: `127.0.0.1`
- Port: `7861`

Override with environment variables before launch if needed:

- `QWEN3_ASR_MODEL`
- `QWEN3_ASR_HF_HOME`
- `QWEN3_ASR_HOST`
- `QWEN3_ASR_PORT`
- `QWEN3_ASR_MAX_NEW_TOKENS`
- `QWEN3_ASR_MAX_BATCH_SIZE`

## First Start

If the model is not cached yet, the first transcription request downloads roughly 1.9 GiB of weights to the SSD-backed cache. Subsequent runs reuse the cached files.