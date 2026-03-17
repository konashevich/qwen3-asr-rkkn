# Qwen3-ASR Rockchip Conversion Workspace

This workspace is aimed at getting Qwen3-ASR onto Rockchip hardware, with RK3588 as the primary target.

## Current Status

The current successful result is a Rockchip-compatible `.rknn` conversion of a reduced Qwen3-ASR speech encoder slice.

- Working artifact: `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn`
- Validation status: successful on PC-side RKNN simulator and compared against ONNX reference output
- Current scope: encoder-only, not full end-to-end ASR transcription yet

This means the project has a real model that can run on RK3588 NPU now, but it is only the speech encoder subgraph. The decoder and multimodal glue are still unresolved.

## Environments

- Main project interpreter: `.venv-qwen3-asr`
- RKNN conversion environment: `.venv-rknn`

Use `.venv-qwen3-asr` as the workspace interpreter in VS Code. The RKNN environment is only for conversion/runtime tooling that has incompatible pins.

## Key Files

- `export_encoder_to_onnx.py`: exports the reduced encoder ONNX model
- `convert_encoder_to_rknn.py`: converts the ONNX artifact into `.rknn`
- `validate_converted.py`: runs the `.rknn` model and compares it with the saved host reference output
- `findings.md`: concise record of what has worked and what is still blocked
- `docs/qwen3-asr-rockchip-conversion-handoff.md`: detailed execution plan and constraints

## What You Can Run On RK3588 Right Now

You can run the reduced encoder model on the board and verify that its output stays close to the host-side ONNX output.

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

## Known Limits

- No full ASR transcription path exists on Rockchip yet
- No RKLLM decoder path has been proven yet
- No real audio sample set has been validated end-to-end in this repo yet
- The current `.rknn` artifact is a reduced single-chunk encoder core, not the complete official Qwen3-ASR model

## Verified Metrics

Host-side checks recorded so far:

- ONNX vs PyTorch for reduced encoder: `max_abs_diff=5.885958671569824e-07`
- RK3588 simulator vs ONNX: `max_abs_diff_vs_onnx=0.0003182440996170044`
- RK3576 simulator vs ONNX: `max_abs_diff_vs_onnx=0.00043520331382751465`

## Next Technical Step

The next step is to determine whether the decoder side can be isolated into a Rockchip-supported path, likely via a hybrid design rather than a single native deployment artifact.