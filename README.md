# Qwen3-ASR Rockchip Conversion Workspace

This workspace is aimed at getting Qwen3-ASR onto Rockchip hardware, with RK3588 as the primary target.

## Current Status

The current validated results are:

- A Rockchip-compatible `.rknn` conversion of a reduced Qwen3-ASR speech encoder slice
- A full board-side Qwen3-ASR-0.6B transcription path running on the CM3588 through the official `qwen-asr` Python runtime



## Environments

- Main project interpreter: `.venv-qwen3-asr`
- RKNN conversion environment: `.venv-rknn`

Use `.venv-qwen3-asr` as the workspace interpreter in VS Code. The RKNN environment is for conversion/runtime tooling with incompatible pins.

## Key Files

- `export_encoder_to_onnx.py`: exports the reduced encoder ONNX model
- `convert_encoder_to_rknn.py`: converts the ONNX artifact into `.rknn`
- `validate_converted.py`: runs the `.rknn` model and compares it with the saved host reference output
- `findings.md`: concise record of the validation phases and recorded results
- `docs/qwen3-asr-rockchip-conversion-handoff.md`: detailed execution plan and constraints

## What You Can Run On RK3588 Right Now

You can run a simple local transcription web app on the board with:

```bash
bash PoC/run_poc.sh
```

Then open:

```text
http://127.0.0.1:7861
```

The PoC uses the official `qwen-asr` runtime and stores Hugging Face model downloads under:

```text
/mnt/merged_ssd/qwen3-asr-model-cache/huggingface
```

You can also run the reduced encoder model packaging flow and inspect the current RKNN deployment state.

### PoC Notes

- First launch downloads `Qwen/Qwen3-ASR-0.6B` if it is not already cached on the SSD.
- A successful on-device smoke test was recorded against `aoede_test.wav`.
- The current PoC is a working transcription application, not a fully native RKNN or RKLLM deployment.

## Reduced Encoder Runtime Bundle

You can package the reduced encoder model for board-side validation with:

The minimal runtime payload is assembled with:

```bash
./scripts/package-rk3588-runtime.sh
```

This creates:

```text
dist/rk3588_encoder_runtime/
```

That folder contains:

- the RK3588 `.rknn` model
- a saved feature tensor input
- the reference encoder output from host ONNX Runtime
- the board-side validator script
- a board-side runtime setup script

Board-side validation on CM3588 now succeeds with the real RKNN artifact and reports:

- `output_shape=[13, 1024]`
- `max_abs_diff=0.0018666512332856655`
- `mean_abs_diff=0.00019976511248387396`

This path is operational for the reduced encoder model on the Rockchip NPU.

On the board:

1. Copy `dist/rk3588_encoder_runtime/` to the device.
2. Place the matching Rockchip `rknn-toolkit-lite2` wheel into that folder.
3. Run the setup script.
4. Run the validator.

Commands:

```bash
bash setup-rk3588-runtime.sh ./rknn_toolkit_lite2-*.whl
./.venv-rk3588-runtime/bin/python validate_converted.py \
  --model qwen3_asr_encoder_single_chunk_rk3588.rknn \
  --input-features input_features.npy \
  --reference-output encoder_reference_output.npy
```

The script writes:

- `device_output.npy`
- `validation_report.json`

## Verified Metrics

Host-side checks recorded so far:

- ONNX vs PyTorch for reduced encoder: `max_abs_diff=5.885958671569824e-07`
- RK3588 simulator vs ONNX: `max_abs_diff_vs_onnx=0.0003182440996170044`
- RK3576 simulator vs ONNX: `max_abs_diff_vs_onnx=0.00043520331382751465`