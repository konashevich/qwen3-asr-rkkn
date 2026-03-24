#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
from rknnlite.api import RKNNLite


DEVICE_COMPATIBLE_NODE = "/proc/device-tree/compatible"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native RKNN encoder on the Rockchip NPU.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-output", default=None)
    parser.add_argument("--core-mask", choices=["auto", "core0", "core1", "core2", "core0_1_2"], default="auto")
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
    return os_machine


def resolve_core_mask(core_mask_name: str) -> Any:
    if core_mask_name == "auto":
        return RKNNLite.NPU_CORE_0 if hasattr(RKNNLite, "NPU_CORE_0") else None
    mapping = {
        "core0": "NPU_CORE_0",
        "core1": "NPU_CORE_1",
        "core2": "NPU_CORE_2",
        "core0_1_2": "NPU_CORE_0_1_2",
    }
    attr_name = mapping[core_mask_name]
    if not hasattr(RKNNLite, attr_name):
        raise RuntimeError(f"Requested core mask {core_mask_name} is not available")
    return getattr(RKNNLite, attr_name)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser().resolve()
    input_features_path = Path(args.input_features).expanduser().resolve()
    reference_output_path = Path(args.reference_output).expanduser().resolve() if args.reference_output else None

    runtime = RKNNLite()
    try:
        ret = runtime.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}")

        init_kwargs: dict[str, Any] = {}
        core_mask = resolve_core_mask(args.core_mask)
        if core_mask is not None:
            init_kwargs["core_mask"] = core_mask

        ret = runtime.init_runtime(**init_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}")

        input_features = np.load(input_features_path).astype(np.float32)
        input_features = np.ascontiguousarray(input_features)
        outputs = runtime.inference(inputs=[input_features])
        device_output = np.array(outputs[0], dtype=np.float32)
        np.save(output_dir / "device_output.npy", device_output)

        validation = None
        if reference_output_path and reference_output_path.exists():
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
            "backend": "rknnlite",
            "host_name": detect_host_name(),
            "model": str(model_path),
            "input_features": str(input_features_path),
            "output_dir": str(output_dir),
            "core_mask": args.core_mask,
            "output_count": len(outputs),
            "output_shape": list(device_output.shape),
            "validation": validation,
            "output_summary": {
                "min": float(device_output.min()),
                "max": float(device_output.max()),
                "mean": float(device_output.mean()),
            },
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "host_name": detect_host_name(),
            "model": str(model_path),
            "input_features": str(input_features_path),
            "output_dir": str(output_dir),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    finally:
        try:
            runtime.release()
        except Exception:
            pass

    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())