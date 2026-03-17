# RKNN Artifact Git LFS Incident Report

## Resolution Status

Status as of 2026-03-18: resolved on `origin/main`.

Verification performed from this producer machine:

- local artifacts are real binaries, not LFS pointer text
- `git lfs status` showed no pending objects to push
- local `main` matched `origin/main`
- a fresh public clone from `https://github.com/konashevich/qwen3-asr-rkkn.git` downloaded the real payloads through Git LFS

Verified fresh-clone artifact sizes:

- `outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx`: about `712M`
- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn`: about `361M`
- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.rknn`: about `369M`
- `dist/rk3588_encoder_runtime/qwen3_asr_encoder_single_chunk_rk3588.rknn`: about `361M`

This means other machines can now clone the public repository and receive the required model binaries from GitHub/LFS.

## Purpose

This report is for the machine that originally exported and converted the Qwen3-ASR encoder artifacts.

The immediate goal is to explain why the current board-side RKNN validation failed and what must be fixed on the producer machine before any further on-device RKNN debugging is meaningful.

## Executive Summary

The current on-device RKNN failure is not evidence that the converted encoder is invalid.

The failure was caused by publishing Git LFS pointer files instead of the real model binaries.

As a result:

- the checked-in `.rknn` files in this repository are plain text pointer files, not RKNN binaries
- the checked-in `.onnx` export is also a plain text pointer file, not a real ONNX model file
- board-side RKNN Lite attempted to parse a text file and reported `invalid RKNN_MAGIC`

This is a publishing / artifact-transfer failure, not yet a proven conversion failure.

## Observed Failure On Device

Board:

- FriendlyElec CM3588
- SoC: RK3588
- Runtime environment already present on board:
  - `/home/pi/npu_env/lib/python3.10/site-packages/rknnlite`
  - `/usr/lib/librknnrt.so`
  - `rknpu` kernel module loaded

Observed RKNN runtime error when validating the encoder artifact:

```text
E RKNN: parseRKNN: invalid RKNN_MAGIC!
E RKNN: parseRKNN from buffer: Invalid RKNN format!
E RKNN: rknn_init, load model failed!
Exception: RKNN init failed. error code: RKNN_ERR_MODEL_INVALID
```

At first glance this looked like a runtime or conversion compatibility problem.

It is not.

## Root Cause

The repository contains Git LFS pointer files instead of the actual model payloads.

### Evidence: RK3588 RKNN file is a pointer, not a binary

File:

- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn`

Observed on this machine:

- `file outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn` reports `ASCII text`
- file size is only `134` bytes

Actual contents:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:6434bea38619c67e94848525cb5ef7a21be29ce06fe6cadf5275704fa74728c0
size 378392962
```

This means the real RKNN model should be about `378,392,962` bytes, but only the LFS pointer text was uploaded or retrieved in this clone.

### Evidence: RK3576 RKNN file is also a pointer

File:

- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.rknn`

Observed on this machine:

- `file ...` reports `ASCII text`
- file size is only `134` bytes

Pointer payload indicates the real file size should be:

- `386,170,818` bytes

### Evidence: ONNX export is also a pointer

File:

- `outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx`

Observed contents:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:273d9b63253c83d93340612090b51e3bc335ed752d26e0b5daea7abab328a019
size 745703406
```

So the ONNX model that should be about `745,703,406` bytes is also missing from this clone.

## Why This Produced `invalid RKNN_MAGIC`

`rknn_toolkit_lite2` expects a valid RKNN binary header.

Instead, it received a text file beginning with:

```text
version https://git-lfs.github.com/spec/v1
```

That is why the board runtime reported:

- `invalid RKNN_MAGIC`
- `Invalid RKNN format`
- `RKNN_ERR_MODEL_INVALID`

This error is exactly what should be expected when passing an LFS pointer into RKNN Lite.

## Additional Context From Metadata

The metadata files in this repository still indicate that real export and conversion were performed elsewhere:

- `outputs/encoder_onnx/export_metadata.json`
- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.json`
- `outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.json`

Those metadata files reference a producer workspace under paths such as:

- `/home/beast/qwen3-asr-rkkn/...`

That strongly suggests the real binary artifacts existed on the producer machine and the failure happened during publish / upload / clone retrieval.

## What Needs To Be Fixed On The Producer Machine

The producer machine should do one of these two things.

### Preferred path: publish the real LFS objects correctly

1. Verify the real files exist on the producer machine and are not pointer text.
2. Verify their sizes approximately match the pointer metadata:
   - RK3588 RKNN: `378,392,962` bytes
   - RK3576 RKNN: `386,170,818` bytes
   - ONNX: `745,703,406` bytes
3. Ensure Git LFS is installed and initialized.
4. Re-add or recommit the real artifacts if necessary.
5. Push both Git commits and LFS objects.
6. On the consumer machine, pull the actual LFS payloads rather than pointer files.

### Fallback path: bypass Git LFS for the transfer

If LFS publishing is unreliable in the current setup, copy the actual `.onnx` and `.rknn` files directly to the target machine by another channel and validate them locally before any more runtime debugging.

## Minimum Verification Checklist On The Producer Machine

Run these checks before publishing again.

### Verify files are real binaries

```bash
file outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx
file outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn
file outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.rknn
ls -lh outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx
ls -lh outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn
ls -lh outputs/rknn/qwen3_asr_encoder_single_chunk_rk3576.rknn
```

None of these should report `ASCII text`.

### Verify Git LFS sees the files as real tracked payloads

```bash
git lfs ls-files
git lfs status
```

### Push the binary objects

```bash
git push
git lfs push --all origin main
```

If this repo uses another branch, replace `main` accordingly.

## Minimum Verification Checklist On The Consumer / Board Side

After re-publishing or copying the real files, re-check:

```bash
file outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn
ls -lh outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn
```

The file must no longer be `ASCII text`.

Only then should board-side validation be retried.

## Important Scope Clarification

Fixing the LFS artifact issue is necessary, but it does not prove full native Qwen3-ASR deployment on Rockchip.

It only restores the ability to test the already-converted reduced encoder artifact correctly.

The bigger unresolved questions remain:

- whether the reduced encoder `.rknn` really initializes and runs on the board once the real binary is present
- whether the decoder path can be made Rockchip-native at all
- whether the multimodal bridge between audio features and text embeddings can be represented in RKLLM or must remain host-side glue

So the correct sequence is:

1. fix LFS / artifact publication
2. validate real encoder `.rknn` on board
3. only then investigate runtime compatibility or architectural native-deployment limits

## Bottom Line

The current `RKNN_ERR_MODEL_INVALID` result should not be used as evidence that the RKNN conversion itself is broken.

The immediate issue is simpler and earlier in the chain:

- the repository currently contains LFS pointer files instead of the actual `.onnx` and `.rknn` payloads

Until the real binaries are restored on the consumer side, any board-side RKNN conclusions are invalid.