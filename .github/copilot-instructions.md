# Copilot Workspace Instructions

## Project scope

This workspace is for Qwen3-ASR Rockchip conversion work on RK3588/RK3576.
The current validated deliverable is a reduced speech-encoder path, not a full end-to-end ASR deployment.

## Default environment

- Use .venv-qwen3-asr for normal project work.
- Use .venv-rknn only for RKNN conversion/runtime tooling.
- Treat Linux x86_64 as the conversion host; the Rockchip board is the runtime target.

## Current state to preserve

- The reduced encoder export and RKNN conversion are already working.
- The decoder and multimodal bridge are still unresolved.
- Keep findings.md and docs/qwen3-asr-rockchip-conversion-handoff.md in sync with any progress.

## Key files

- README.md: current status and runtime notes
- findings.md: phase-by-phase status tracker
- docs/qwen3-asr-rockchip-conversion-handoff.md: detailed execution plan and constraints
- inspect_model.py: model structure inspection
- export_encoder_to_onnx.py: reduced encoder ONNX export
- convert_encoder_to_rknn.py: RKNN conversion
- validate_baseline.py: baseline Qwen3-ASR validation
- validate_converted.py: converted-model comparison
- scripts/: environment setup and runtime packaging helpers

## Build, validation, and runtime commands

- Setup the main environment with scripts/setup-qwen3-asr-venv.sh.
- Inspect the model with inspect_model.py.
- Run baseline validation with validate_baseline.py.
- Export ONNX with export_encoder_to_onnx.py.
- Convert to RKNN with convert_encoder_to_rknn.py.
- Validate simulator or device output with validate_converted.py.
- Package the RK3588 runtime with scripts/package-rk3588-runtime.sh.
- Prepare the board runtime with scripts/setup-rk3588-runtime.sh.

## Working conventions

- Prefer direct replacements over compatibility shims or fallback paths unless explicitly requested.
- Keep changes minimal and aligned with the existing phase-based workflow.
- Update documentation when behavior, scope, or status changes.
- Avoid implying full model support when only the encoder path has been validated.

## Common pitfalls

- Do not assume the full model can be converted as a single graph.
- The current encoder export uses a reduced single-chunk path because full chunk padding export is not practical.
- RKNN and RKLLM are separate toolchains; do not mix their assumptions.
