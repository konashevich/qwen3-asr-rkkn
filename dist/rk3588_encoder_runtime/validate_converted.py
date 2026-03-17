#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np


DEVICE_COMPATIBLE_NODE = "/proc/device-tree/compatible"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the converted Qwen3-ASR encoder RKNN model and optionally compare it with a host-side reference output."
    )
    parser.add_argument(
        "--model",
        default="outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn",
        help="Path to the RKNN model to run on the device.",
    )
    parser.add_argument(
        "--input-features",
        default="outputs/runtime_reference/input_features.npy",
        help="Path to the input feature tensor saved as a NumPy .npy file.",
    )
    parser.add_argument(
        "--reference-output",
        default="outputs/runtime_reference/encoder_reference_output.npy",
        help="Optional host-side reference output to compare against.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/device_run",
        help="Directory where the runtime output and validation report will be written.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional Rockchip target name for remote runtime via RKNN Toolkit APIs.",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="Optional device id for remote runtime via RKNN Toolkit APIs.",
    )
    parser.add_argument(
        "--core-mask",
        choices=["auto", "core0", "core1", "core2", "core0_1_2"],
        default="auto",
        help="Preferred NPU core selection when running with RKNNLite on RK3588 or RK3576.",
    )
    return parser.parse_args()


def detect_host_name() -> str:
    system = platform.system()
    machine = platform.machine()
    os_machine = f"{system}-{machine}"
    if os_machine != "Linux-aarch64":
        return os_machine

    try:
        compatible = Path(DEVICE_COMPATIBLE_NODE).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return os_machine

    if "rk3588" in compatible:
        return "RK3588"
    if "rk3576" in compatible:
        return "RK3576"
    if "rk3562" in compatible:
        return "RK3562"
    return "RK3566_RK3568"


def resolve_core_mask(runtime: Any, core_mask_name: str, host_name: str) -> Any:
    if core_mask_name == "auto":
        if host_name in {"RK3588", "RK3576"} and hasattr(runtime, "NPU_CORE_0"):
            return runtime.NPU_CORE_0
        return None

    mapping = {
        "core0": "NPU_CORE_0",
        "core1": "NPU_CORE_1",
        "core2": "NPU_CORE_2",
        "core0_1_2": "NPU_CORE_0_1_2",
    }
    attr_name = mapping[core_mask_name]
    if not hasattr(runtime, attr_name):
        raise RuntimeError(f"Requested core mask {core_mask_name} is not available in this RKNN runtime.")
    return getattr(runtime, attr_name)


def init_rknn_runtime(model_path: Path, target: str | None, device_id: str | None, core_mask_name: str) -> tuple[Any, str, str, Any]:
    host_name = detect_host_name()

    try:
        from rknnlite.api import RKNNLite  # type: ignore

        runtime = RKNNLite()
        ret = runtime.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}")

        init_kwargs: dict[str, Any] = {}
        core_mask = resolve_core_mask(RKNNLite, core_mask_name, host_name)
        if core_mask is not None:
            init_kwargs["core_mask"] = core_mask

        ret = runtime.init_runtime(**init_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}")
        return runtime, "rknnlite", host_name, core_mask_name if core_mask is not None else "none"
    except ImportError:
        if not target:
            raise RuntimeError(
                "rknnlite.api is not available in this Python environment. "
                "Run this script directly on the RK3588/RK3576 board with rknn-toolkit-lite2 installed, "
                "or provide --target and optionally --device-id to execute against a connected board via RKNN Toolkit. "
                "A loaded .rknn model cannot be simulated on x86 with RKNN.load_rknn()."
            )

        from rknn.api import RKNN  # type: ignore

        runtime = RKNN(verbose=False)
        ret = runtime.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN.load_rknn failed with code {ret}")

        init_kwargs: dict[str, Any] = {}
        if target:
            init_kwargs["target"] = target
        if device_id:
            init_kwargs["device_id"] = device_id

        ret = runtime.init_runtime(**init_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN.init_runtime failed with code {ret}")
        return runtime, "rknn", host_name, "n/a"


def release_runtime(runtime: Any) -> None:
    try:
        runtime.release()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    input_features_path = Path(args.input_features).expanduser().resolve()
    reference_output_path = Path(args.reference_output).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_features = np.load(input_features_path).astype(np.float32)
    input_features = np.ascontiguousarray(input_features)

    try:
        runtime, backend_name, host_name, used_core_mask = init_rknn_runtime(
            model_path=model_path,
            target=args.target,
            device_id=args.device_id,
            core_mask_name=args.core_mask,
        )
        outputs = runtime.inference(inputs=[input_features])
        device_output = np.array(outputs[0], dtype=np.float32)
    except Exception as exc:
        report = {
            "status": "failed",
            "host_name": detect_host_name(),
            "model": str(model_path),
            "input_features": str(input_features_path),
            "output_dir": str(output_dir),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        (output_dir / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    finally:
        if "runtime" in locals():
            release_runtime(runtime)

    np.save(output_dir / "device_output.npy", device_output)

    validation: dict[str, Any] | None = None
    if reference_output_path.exists():
        reference_output = np.load(reference_output_path).astype(np.float32)
        abs_diff = np.abs(device_output - reference_output)
        validation = {
            "reference_output": str(reference_output_path),
            "reference_shape": list(reference_output.shape),
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
        }

    report = {
        "status": "ok",
        "backend": backend_name,
        "host_name": host_name,
        "model": str(model_path),
        "input_features": str(input_features_path),
        "output_dir": str(output_dir),
        "core_mask": used_core_mask,
        "output_count": len(outputs),
        "output_shape": list(device_output.shape),
        "validation": validation,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())