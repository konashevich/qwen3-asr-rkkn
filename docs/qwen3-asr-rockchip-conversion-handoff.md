# Qwen3-ASR to Rockchip Conversion Handoff Specification

## Purpose

This document is a handoff specification for an AI agent running on a separate, more powerful machine. The agent's job is to determine whether Qwen3-ASR can be converted into a Rockchip-native deployment on RK3588 or RK3576, and if so, to produce the converted artifacts and a reproducible workflow.

This is not a request for a generic summary. It is an execution plan.

The agent must follow the steps below, record findings at each gate, and stop when a gate fails rather than wasting time forcing an unsupported path.

## Executive Summary

The most important conclusion from prior research is this:

1. Qwen3-ASR is not a plain text LLM. It is a composite speech model with its own audio-oriented architecture and official runtime stack.
2. Rockchip provides two separate official toolchains:
   - RKNN-Toolkit2 for regular neural network conversion to `.rknn`
   - RKNN-LLM for supported text and multimodal LLM conversion to `.rkllm`
3. There is no public evidence of an official or ready-made native RKNN conversion path for Qwen3-ASR.
4. Public searches for `Qwen3-ASR Rockchip`, `Qwen3-ASR rknn`, and `Qwen3-ASR onnx` did not reveal a working public Rockchip port.
5. Unofficial local deployment work exists for GGUF and llama.cpp style reduced runtimes, but that is not the same as a native Rockchip conversion.
6. The correct conversion host is Linux x86_64. Do not use ARM64 as the conversion machine. Windows x86_64 is possible for some tooling, but Linux x86_64 is the preferred and practical platform.

## Required Host Platform

Use this host platform unless there is a hard constraint:

- OS: Ubuntu 22.04 x86_64 preferred, Ubuntu 20.04 x86_64 acceptable
- CPU: x86_64 only for conversion work
- RAM: 64 GB minimum, 96 GB preferred
- GPU: NVIDIA CUDA GPU strongly preferred for baseline model validation and faster export/debug; not strictly required for every step, but highly recommended
- Disk: At least 100 GB free

Do not use these as the main conversion host:

- ARM64 Linux
- Raspberry Pi
- Rockchip board itself
- Windows as the primary path

The Rockchip board is the runtime target, not the conversion workstation.

## Target Platforms

Primary targets:

- RK3588
- RK3576

The target board should only be used after host-side conversion and validation are complete.

## Official Repositories and Toolchains

These are the official repos and roles that matter.

### Rockchip

1. `https://github.com/airockchip/rknn-toolkit2`

Purpose:

- Convert regular models to `.rknn`
- Includes `rknn-toolkit2`, `rknn-toolkit-lite2`, and `rknpu2`

Relevant findings:

- Officially described as the PC-side conversion toolkit
- Outputs RKNN models for inference on supported Rockchip NPUs
- Supports RK3588, RK3576, RK3566, RK3568, RK3562, RV1103, RV1106, RV1126B, RK2118
- Current official release observed: v2.3.2

2. `https://github.com/airockchip/rknn_model_zoo`

Purpose:

- Reference models, export examples, deployment patterns, and sample pipelines

Relevant finding:

- This is where Rockchip’s example conversions and runtime wiring patterns live

3. `https://github.com/airockchip/rknn-llm`

Purpose:

- Convert supported LLMs to `.rkllm`
- Provide RKLLM runtime for deployment on supported Rockchip NPUs

Relevant findings:

- Officially described as the PC-side conversion and quantization toolkit for LLMs
- Supports RK3588, RK3576, RK3562, RV1126B
- Supports families such as LLaMA, TinyLLaMA, Qwen2, Qwen2.5, Qwen3, Phi, Gemma, InternLM2, MiniCPM, and others
- Current official release observed: v1.2.3

4. `https://github.com/rockchip-linux/rknpu2`

Purpose:

- Older public runtime/API repository

Relevant finding:

- Its own README states the active maintained path moved under `airockchip/rknn-toolkit2/tree/master/rknpu2`

### Qwen

5. `https://github.com/QwenLM/Qwen3-ASR`

Purpose:

- Official code and inference framework for Qwen3-ASR

Relevant findings:

- Release date observed: 2026-01-29
- Public model sizes: 0.6B and 1.7B
- Official package: `qwen-asr`
- Supports a transformers backend and a vLLM backend
- Streaming inference is only officially available through the vLLM backend
- Separate official forced aligner exists: `Qwen3-ForcedAligner-0.6B`
- The repo presents a dedicated model architecture and a dedicated speech inference framework, which strongly suggests a composite speech model rather than a plain text LLM

## Non-Official Reference Repositories

These are useful only as references for model splitting and local reduced-format deployment ideas. They are not evidence of a proper Rockchip-native path.

1. `https://github.com/HaujetZhao/Qwen3-ASR-GGUF`

Observed role:

- Hybrid export path using ONNX for some components and GGUF/llama.cpp for others

2. `https://github.com/shershah1024/qwen3-asr-llamacpp`

Observed role:

- Adds Qwen3-ASR support to llama.cpp with custom patches and conversion steps

Interpretation:

- These projects show that the model often needs to be split and adapted rather than converted whole
- They do not prove native RKNN or RKLLM support

## Known Facts About Qwen3-ASR

The next agent must begin with these assumptions unless direct code inspection disproves them.

1. Qwen3-ASR is a speech model with audio input support, language identification, transcription, and optional forced alignment.
2. Its official usage relies on `qwen-asr` and either transformers or vLLM.
3. The official docs recommend Python 3.12 for the Qwen3-ASR environment.
4. vLLM is the preferred official backend for fastest inference and for streaming support.
5. The model is almost certainly not equivalent to taking a standard Qwen text checkpoint and attaching a trivial audio front end.
6. Because Rockchip splits support between RKNN and RKLLM, full native deployment will likely require separating the speech model into components.

## Core Hypothesis to Test

The working hypothesis is:

Qwen3-ASR may only be deployable on Rockchip as a hybrid pipeline, not as a single fully native Rockchip model.

Possible hybrid outcome:

- CPU: audio loading, resampling, feature extraction, or unsupported glue logic
- RKNN: encoder or encoder subgraphs if export succeeds
- RKLLM: decoder only if it is compatible with Rockchip’s supported Qwen-family LLM conversion path
- CPU or custom runtime: remaining unsupported cross-modal logic

The job of the next agent is to confirm or reject this hypothesis with evidence.

## Success Criteria

The job counts as fully successful only if all of the following are achieved:

1. A documented and reproducible conversion path exists on Linux x86_64.
2. At least one Qwen3-ASR variant, preferably 0.6B, is converted into deployable Rockchip artifacts.
3. The runtime on RK3588 or RK3576 can accept real audio input and produce correct transcription.
4. Accuracy is close enough to the original model to be useful.
5. The entire pipeline is scripted and repeatable.

The job counts as partially successful if:

1. The model is split into deployable components.
2. One or more components convert successfully.
3. A hybrid CPU + RKNN and/or CPU + RKLLM path runs end to end.

The job counts as unsuccessful if:

1. The model cannot be exported cleanly to ONNX in usable subgraphs.
2. RKNN conversion fails due to unsupported operators or graph structure with no practical workaround.
3. The decoder cannot be converted to RKLLM or reused as a supported Qwen-family LLM.

## Hard Constraints and Stop Conditions

The next agent must not grind through endless trial-and-error. Use these stop conditions.

Stop Condition A:

- If the encoder cannot be exported to ONNX in a runnable form, stop the native Rockchip path and report that the architecture is not practically convertible through the standard Rockchip flow.

Stop Condition B:

- If the encoder exports to ONNX but cannot be converted to RKNN due to unsupported operators or dynamic graph requirements that cannot be removed without major model surgery, stop and report that only a non-native or heavily hybrid solution is practical.

Stop Condition C:

- If the decoder depends on cross-modal internals that RKLLM cannot represent, stop the full-native plan and report that only partial acceleration is possible.

Stop Condition D:

- If the converted output is materially incorrect compared with baseline Qwen3-ASR on the same sample set, stop and report the regression.

## Required Deliverables

The next agent must produce all of the following.

1. A written report with exact findings.
2. A conversion workspace containing scripts.
3. A baseline validation run against the official Qwen3-ASR implementation.
4. Any generated `.onnx`, `.rknn`, and `.rkllm` artifacts.
5. A runtime demo or CLI that takes audio and returns transcription.
6. A failure report if the native path does not work.

Minimum files expected from the next agent:

- `README.md`
- `env-qwen3-asr.yml` or equivalent environment setup
- `env-rknn.txt` or equivalent environment setup
- `inspect_model.py`
- `export_encoder_to_onnx.py`
- `export_projector_to_onnx.py` if applicable
- `convert_encoder_to_rknn.py`
- `convert_decoder_to_rkllm.md` or automation if feasible
- `validate_baseline.py`
- `validate_converted.py`
- `samples/` with test audio and expected text
- `findings.md`

## Mandatory Workflow

Follow the workflow exactly in order.

### Phase 0: Prepare the Host Environment

Use Linux x86_64.

Create separate environments to avoid dependency conflicts.

Recommended environments:

1. `qwen3-asr` environment
   - Python 3.12
   - `qwen-asr`
   - optional `qwen-asr[vllm]`
   - `torch`
   - `transformers`
   - `huggingface_hub`
   - `modelscope` if needed
   - `onnx`
   - `onnxruntime`
   - `onnxsim`

2. `rockchip-rknn` environment
   - Python version compatible with the chosen RKNN Toolkit2 build
   - `rknn-toolkit2`
   - `rknn-toolkit-lite2` if needed for board-side testing

3. `rockchip-rkllm` environment
   - Python 3.9 to 3.12, according to the chosen RKLLM build
   - `rkllm-toolkit`

Recommendation:

- Use Docker or isolated virtual environments
- Keep Qwen, RKNN, and RKLLM toolchains separate

### Phase 1: Baseline Official Inference

Goal:

- Prove the official Qwen3-ASR model runs correctly before touching conversion

Actions:

1. Clone `QwenLM/Qwen3-ASR`.
2. Install the official package or editable source.
3. Download at least `Qwen/Qwen3-ASR-0.6B`.
4. Run baseline transcription on a small fixed sample set.
5. Save outputs as the gold reference.

Sample set requirements:

- At least 5 short Chinese clips
- At least 3 English clips
- At least 2 longer clips
- At least 1 noisy clip

Save:

- raw model output
- parsed text output
- timing and runtime notes

This phase is mandatory. Do not skip it.

### Phase 2: Inspect the Actual Model Structure

Goal:

- Identify the real module boundaries for export

Actions:

1. Inspect `qwen_asr` source code and the loaded model object.
2. Enumerate major submodules.
3. Determine which pieces correspond to:
   - audio preprocessing
   - speech encoder
   - projector or adapter layers
   - decoder or LLM core
   - post-processing and generation logic
4. Determine whether audio preprocessing is in Python code, model graph, or both.
5. Determine the exact shapes and datatypes flowing between components.

Deliverable for this phase:

- A diagram or markdown section showing the component split

This phase determines whether Rockchip conversion is plausible.

### Phase 3: Export Candidate Subgraphs to ONNX

Goal:

- Find the largest exportable subgraph that still maps cleanly to Rockchip

Order of attempts:

1. Export speech encoder only
2. Export projector or connector only if present
3. Export combined encoder + projector if separate export is too fragmented
4. Do not attempt the full model as one first step

Requirements:

- Prefer static shapes first
- Use short fixed-length audio windows first
- Record operator inventory after export

Validation requirements:

1. Run the exported ONNX in ONNX Runtime.
2. Compare tensor outputs to the original PyTorch model.
3. Only continue if outputs are acceptably close.

If ONNX Runtime validation fails, stop the Rockchip path and report the blocker.

### Phase 4: Simplify and Prepare the ONNX Graph

Goal:

- Make the ONNX graph more likely to pass RKNN conversion

Actions:

1. Run ONNX simplification.
2. Freeze dynamic shapes if possible.
3. Remove unsupported graph patterns when safe.
4. Prefer smaller sequence lengths for the first successful proof.
5. Generate an operator inventory and highlight anything suspicious:
   - unusual attention variants
   - custom ops
   - control-flow ops
   - audio-domain transforms embedded in the graph

Deliverable:

- a simplified ONNX file and a short note explaining what changed

### Phase 5: Attempt RKNN Conversion

Goal:

- Convert the validated ONNX encoder path to `.rknn`

Actions:

1. Use `rknn-toolkit2` on Linux x86_64.
2. Target RK3588 first.
3. If RK3588 succeeds, test whether RK3576 also works.
4. Start with the smallest viable exported component.

Requirements:

- Record exact toolkit version
- Record preprocessing configuration
- Record quantization settings
- Record whether the conversion is static-shape or dynamic-shape

Validation:

1. Run inference with RKNN simulator if available.
2. Run inference on real board.
3. Compare RKNN outputs against ONNX and PyTorch outputs.

If accuracy is unacceptable or conversion fails on unsupported operators, stop and report.

### Phase 6: Evaluate Decoder Feasibility for RKLLM

Goal:

- Determine whether the decoder side can be mapped to RKLLM

Important warning:

Do not assume this will work just because `rknn-llm` supports Qwen-family text models.

Actions:

1. Determine whether Qwen3-ASR’s decoder is structurally close enough to a supported Qwen text model.
2. Determine what inputs the decoder expects from the speech side.
3. Determine whether those inputs can be reproduced in a way RKLLM can consume.
4. Check whether the generation path requires custom multimodal bridging not supported by RKLLM.

Possible outcomes:

1. Decoder is convertible to `.rkllm`
2. Decoder is only usable outside RKLLM
3. Decoder cannot be separated cleanly enough for practical deployment

If the decoder cannot be represented in RKLLM, report that full-native deployment is not feasible.

### Phase 7: Build the Minimal End-to-End Prototype

Goal:

- Build the simplest working runtime that proves what is and is not possible

Prototype priority order:

1. Best case: full or mostly native Rockchip path
2. Next best: hybrid CPU + RKNN path
3. Next best: hybrid CPU + RKNN + RKLLM path
4. Final fallback: report failure and recommend a different model

The runtime must:

- accept a local audio file
- resample if needed
- run through the converted or hybrid pipeline
- emit plain text transcription

### Phase 8: Compare Against Baseline

Goal:

- Determine whether the converted pipeline is practically useful

Metrics:

- qualitative transcription comparison
- runtime on board
- memory usage
- NPU usage if measurable
- failure cases

If the converted pipeline is much worse than baseline or too fragile, document that clearly.

## Practical Decision Tree

Use this exact decision tree.

1. Can Qwen3-ASR-0.6B run correctly in the official environment?
   - If no: stop and fix the baseline environment first.
   - If yes: continue.

2. Can the encoder be exported to ONNX and validated in ONNX Runtime?
   - If no: stop the native Rockchip plan.
   - If yes: continue.

3. Can the encoder ONNX be converted to RKNN with acceptable output fidelity?
   - If no: stop the native RKNN encoder plan.
   - If yes: continue.

4. Can the decoder be isolated and mapped to RKLLM without losing required cross-modal behavior?
   - If no: report that only partial acceleration is feasible.
   - If yes: continue.

5. Can the full end-to-end system transcribe real audio correctly on Rockchip?
   - If no: report the partial result and stop.
   - If yes: deliver the full workflow.

## Suggested Technical Approach

The most realistic order of attack is:

1. Use Qwen3-ASR-0.6B, not 1.7B, as the first target.
2. Ignore streaming in the first conversion attempt.
3. Ignore the forced aligner in the first conversion attempt.
4. Treat the model as multiple pieces, not one monolith.
5. Prove the encoder path first.
6. Only then investigate whether the decoder can be mapped to RKLLM.

Why:

- Smaller model means faster iteration and lower memory pressure.
- Streaming introduces extra runtime complexity.
- Forced alignment is a separate model and should not be mixed into the first proof.
- Rockchip toolchains are designed for converting well-defined subgraphs, not arbitrary composite systems in one shot.

## What the Agent Must Not Do

1. Do not attempt the main conversion work on ARM64.
2. Do not begin with Windows unless Linux is truly unavailable.
3. Do not assume `Qwen3` support in RKLLM automatically means `Qwen3-ASR` support.
4. Do not waste time trying to convert the entire model as a single first attempt.
5. Do not declare success unless transcription works end to end on actual target hardware.
6. Do not ignore accuracy drift between baseline and converted outputs.

## Suggested Environment Setup Skeleton

This is a sketch, not a final locked recipe.

### Qwen baseline environment

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U qwen-asr
pip install -U "huggingface_hub[cli]" onnx onnxruntime onnxsim
git clone https://github.com/QwenLM/Qwen3-ASR.git
cd Qwen3-ASR
pip install -e .
```

Optional faster path if GPU is available:

```bash
pip install -U qwen-asr[vllm]
```

### RKNN environment

Use the official Rockchip package version matching the chosen platform and host environment. Install it in a separate environment from Qwen.

### RKLLM environment

Use the official `rknn-llm` package version matching the chosen platform and host environment. Install it in a separate environment from both Qwen and RKNN.

## Suggested Investigation Tasks for the Agent

The next agent should complete these concrete tasks.

1. Create a local workspace for the job.
2. Run official Qwen3-ASR baseline inference on a fixed sample set.
3. Inspect the model object and list component boundaries.
4. Export the encoder to ONNX.
5. Validate encoder ONNX numerically against PyTorch.
6. Simplify and static-shape the ONNX graph.
7. Attempt RKNN conversion of the encoder.
8. Validate RKNN output numerically.
9. Investigate decoder-to-RKLLM feasibility.
10. Build a minimal end-to-end prototype.
11. Measure correctness and speed.
12. Write the final report.

## Exact Questions the Agent Must Answer

The final report must answer these questions clearly.

1. Is Qwen3-ASR convertible to a fully native Rockchip deployment?
2. If not fully native, what hybrid split is actually possible?
3. Which exact components run on CPU, RKNN, and RKLLM?
4. Which model variant was tested?
5. Which Rockchip chip was targeted?
6. Which host platform and toolkit versions were used?
7. Which operator or graph issues blocked conversion, if any?
8. What is the measured transcription quality compared with baseline?
9. What is the measured runtime on target hardware?
10. Is the result good enough to be worth using?

## Expected Final Recommendation Logic

The final recommendation should follow this logic.

1. If full native conversion works, provide the exact scripts and artifacts.
2. If only partial acceleration works, recommend the hybrid deployment and explain why.
3. If Qwen3-ASR is not practical for Rockchip, say so directly and recommend switching to a smaller ASR model with a more standard encoder path and better RKNN compatibility.

## Bottom Line

The next agent should proceed under this assumption:

The likely path is not a clean one-shot conversion of Qwen3-ASR into a single Rockchip-native model. The practical path is to split the model, prove the encoder first through ONNX to RKNN, then separately test whether the decoder can be represented in RKLLM. If either half fails at a hard gate, the job should stop and the report should clearly state that Qwen3-ASR is not presently a practical full-native Rockchip target.