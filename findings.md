# Findings

## Phase 1 Baseline

- Status: not run yet
- Model:
- Environment:
- Sample set:
- Notes:

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
  - This is only the reduced speech encoder core, not an end-to-end Qwen3-ASR deployment.

## Phase 6 RKLLM Feasibility

- Status: not started
- Notes: