#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the reduced Qwen3-ASR encoder ONNX model to RKNN and optionally validate with the PC simulator."
    )
    parser.add_argument(
        "--onnx-model",
        default="outputs/encoder_onnx/qwen3_asr_encoder_single_chunk.onnx",
        help="Path to the ONNX model produced by export_encoder_to_onnx.py.",
    )
    parser.add_argument(
        "--sample-inputs",
        default="outputs/encoder_onnx/sample_inputs.npz",
        help="NPZ file containing input_features and feature_lens for validation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rknn",
        help="Directory for the exported RKNN model and metadata.",
    )
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        choices=["rk3588", "rk3576"],
        help="Rockchip target platform.",
    )
    parser.add_argument(
        "--do-quantization",
        action="store_true",
        help="Enable quantization during the RKNN build step.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Quantization dataset file. Required if --do-quantization is used.",
    )
    parser.add_argument(
        "--skip-simulator",
        action="store_true",
        help="Skip PC simulator inference after conversion.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RKNN logging.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_onnx_input_names(path: Path) -> list[str]:
    model = onnx.load(str(path))
    return [value.name for value in model.graph.input]


def run_onnx_reference(path: Path, input_features: np.ndarray, feature_lens: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    feed_dict: dict[str, Any] = {"input_features": input_features}
    if "feature_lens" in input_names:
        feed_dict["feature_lens"] = feature_lens
    outputs = session.run(None, feed_dict)
    return np.array(outputs[0])


def main() -> int:
    args = parse_args()
    if args.do_quantization and not args.dataset:
        raise ValueError("--dataset is required when --do-quantization is enabled.")

    onnx_model = Path(args.onnx_model).expanduser().resolve()
    sample_inputs = Path(args.sample_inputs).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rknn_path = output_dir / f"qwen3_asr_encoder_single_chunk_{args.target_platform}.rknn"
    metadata_path = output_dir / f"qwen3_asr_encoder_single_chunk_{args.target_platform}.json"

    samples = np.load(sample_inputs)
    input_features = samples["input_features"].astype(np.float32)
    feature_lens = samples["feature_lens"].astype(np.int64)
    onnx_input_names = get_onnx_input_names(onnx_model)

    rknn = RKNN(verbose=args.verbose)
    status = "ok"
    error = None
    simulator = None

    try:
        ret = rknn.config(target_platform=args.target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(onnx_model))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        ret = rknn.build(do_quantization=args.do_quantization, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")

        if not args.skip_simulator:
            ret = rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"rknn.init_runtime failed with code {ret}")

            simulator_inputs = [input_features]
            if len(onnx_input_names) > 1 and any(name == "feature_lens" for name in onnx_input_names):
                simulator_inputs.append(feature_lens)
            outputs = rknn.inference(inputs=simulator_inputs)
            simulator_output = np.array(outputs[0])
            onnx_output = run_onnx_reference(onnx_model, input_features, feature_lens)
            abs_diff = np.abs(simulator_output - onnx_output)
            simulator = {
                "onnx_input_names": onnx_input_names,
                "output_count": len(outputs),
                "output_shapes": [list(np.array(output).shape) for output in outputs],
                "max_abs_diff_vs_onnx": float(abs_diff.max()),
                "mean_abs_diff_vs_onnx": float(abs_diff.mean()),
            }

    except Exception as exc:
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        try:
            rknn.release()
        except Exception:
            pass

    metadata = {
        "status": status,
        "target_platform": args.target_platform,
        "onnx_model": str(onnx_model),
        "sample_inputs": str(sample_inputs),
        "rknn_model": str(rknn_path),
        "do_quantization": args.do_quantization,
        "dataset": args.dataset,
        "onnx_input_names": onnx_input_names,
        "simulator": simulator,
        "error": error,
    }
    write_json(metadata_path, metadata)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())