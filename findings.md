# Findings

## Phase 1 Baseline

- Status: completed on CM3588 for single-sample board validation
- Model: `Qwen/Qwen3-ASR-0.6B`
- Environment: workspace `.venv` on Linux aarch64 CM3588, with Hugging Face cache redirected to `/mnt/merged_ssd/qwen3-asr-model-cache/huggingface`
- Sample set:
  - `/mnt/merged_ssd/app_data/openclaw_data/workspace/aoede_test.wav`
- Notes:
  - Full `qwen-asr` inference runs on the board and produced a correct English transcript.
  - First model load took `159.01s` including initial weight download.
  - The sample transcription took `92.55s` after model load.
  - Output was written under `outputs/poc_smoke/`.

## Phase 2 Model Inspection

- Status: completed for config-level inspection
- Evidence:
  - `inspect_model.py`
  - `outputs/inspect/model_inspection.json`
  - `outputs/inspect/model_inspection.md`
- Preliminary notes:
  - Python-side audio preprocessing and chunking happen outside the model graph.
  - The speech encoder candidate is `thinker.audio_tower`.
  - The text decoder candidate is `thinker.model` plus `thinker.lm_head`.
  - The multimodal bridge inserts audio features into text embeddings, which may block direct RKLLM reuse.
  - For `Qwen/Qwen3-ASR-0.6B`, the audio encoder config is 18 layers with 128 mel bins and 1024 output dim.
  - For `Qwen/Qwen3-ASR-0.6B`, the decoder config is 28 layers with hidden size 1024 and vocab size 151936.

## Phase 3 ONNX Export

- Status: reduced encoder export succeeded
- Candidate target: `thinker.audio_tower`
- Blockers:
  - Static-shape export attempt from `export_encoder_to_onnx.py` fails on unsupported ONNX export of `aten::pad_sequence`.
  - The failure occurs inside `Qwen3ASRAudioEncoder.forward()` during chunk padding, before any RKNN conversion step.
  - Workaround used: export a reduced single-chunk encoder core with chunk padding moved out of the graph.
  - Result: `outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx` was accepted by RKNN conversion.
  - Validation: ONNX Runtime matches the PyTorch reference for the reduced encoder with `max_abs_diff=5.885958671569824e-07` and `mean_abs_diff=7.435717463977198e-08`.

## Phase 4 ONNX Simplification

- Status: not started
- Notes:

## Phase 5 RKNN Conversion

- Status: reduced encoder conversion succeeded for RK3588 and RK3576
- Notes:
  - `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn` built successfully and ran in the RKNN PC simulator.
  - RK3588 simulator output matched ONNX with `max_abs_diff_vs_onnx=0.0003182440996170044` and `mean_abs_diff_vs_onnx=3.526877480908297e-05`.
  - `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.rknn` built successfully and ran in the RKNN PC simulator.
  - RK3576 simulator output matched ONNX with `max_abs_diff_vs_onnx=0.00043520331382751465` and `mean_abs_diff_vs_onnx=4.207476376905106e-05`.
  - Board-side runtime on CM3588 is now validated after pulling the real Git LFS payloads into this workspace.
  - `validate_converted.py` under `/home/pi/npu_env` completed successfully with backend `rknnlite`.
  - Board-side validation reported `output_shape=[13, 1024]`, `max_abs_diff=0.0018666512332856655`, and `mean_abs_diff=0.00019976511248387396` versus the saved reference output.
  - This is only the reduced speech encoder core, not an end-to-end Qwen3-ASR deployment.

## Phase 7 Board PoC

- Status: completed for a local Flask-based transcription PoC
- Evidence:
  - `PoC/app.py`
  - `PoC/templates/index.html`
  - `PoC/static/app.js`
  - `PoC/static/style.css`
  - `PoC/run_poc.sh`
- Notes:
  - The PoC runs on the CM3588 board using the official `qwen-asr` package and `Qwen/Qwen3-ASR-0.6B`.
  - Browser upload and browser microphone recording are supported.
  - SSD-backed model cache is the default to avoid exhausting the root filesystem.

## Phase 6 RKLLM Feasibility

- Status: not started
- Notes: