#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("RKNN_POC_HOST", "127.0.0.1")
PORT = int(os.environ.get("RKNN_POC_PORT", "7861"))
RKNN_PYTHON = Path(os.environ.get("RKNN_POC_RKNN_PYTHON", "/home/pi/npu_env/bin/python"))
RKNN_MODEL_PATH = WORKSPACE_DIR / "outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn"
REFERENCE_INPUT_PATH = WORKSPACE_DIR / "outputs/runtime_reference/input_features.npy"
REFERENCE_OUTPUT_PATH = WORKSPACE_DIR / "outputs/runtime_reference/encoder_reference_output.npy"
REPORT_DIR = WORKSPACE_DIR / "outputs/device_run_on_board"
NATIVE_HELPER = BASE_DIR / "run_native_encoder.py"
ALLOWED_EXTENSIONS = {".npy"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _module_loaded(name: str) -> bool:
    modules = _read_text(Path("/proc/modules")) or ""
    return any(line.startswith(f"{name} ") for line in modules.splitlines())


def _disk_summary(path: Path) -> dict[str, Any]:
    usage = os.statvfs(path)
    total = usage.f_blocks * usage.f_frsize
    free = usage.f_bavail * usage.f_frsize
    used = total - (usage.f_bfree * usage.f_frsize)
    gib = 1024 ** 3
    return {
        "path": str(path),
        "total_gib": round(total / gib, 2),
        "used_gib": round(used / gib, 2),
        "free_gib": round(free / gib, 2),
    }


def _model_size_mb(path: Path) -> float | None:
    try:
        return round(path.stat().st_size / (1024 ** 2), 2)
    except OSError:
        return None


def run_encoder(input_path: Path, output_dir: Path, reference_output: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(RKNN_PYTHON),
        str(NATIVE_HELPER),
        "--model",
        str(RKNN_MODEL_PATH),
        "--input-features",
        str(input_path),
        "--output-dir",
        str(output_dir),
    ]
    if reference_output is not None:
        command.extend(["--reference-output", str(reference_output)])

    result = subprocess.run(
        command,
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    report = _read_json(output_dir / "validation_report.json")
    if report is None:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Native encoder run did not produce a report")
    report["command"] = command
    report["stdout"] = result.stdout.strip()
    report["stderr"] = result.stderr.strip()
    return report


def collect_diagnostics() -> dict[str, Any]:
    board_model = _read_text(Path("/proc/device-tree/model"))
    compatible_text = _read_text(Path("/proc/device-tree/compatible")) or ""
    compatible = [item for item in compatible_text.replace("\x00", "\n").splitlines() if item]
    board_report = _read_json(REPORT_DIR / "validation_report.json")
    return {
        "board_model": board_model,
        "compatible": compatible,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "npu": {
            "rknpu_module_loaded": _module_loaded("rknpu"),
            "librknnrt_present": Path("/usr/lib/librknnrt.so").exists(),
            "rknnlite_site_present": Path("/home/pi/npu_env/lib/python3.10/site-packages/rknnlite").exists(),
            "rknn_python": str(RKNN_PYTHON),
            "encoder_rknn_path": str(RKNN_MODEL_PATH),
            "encoder_rknn_size_mb": _model_size_mb(RKNN_MODEL_PATH),
            "reference_input_path": str(REFERENCE_INPUT_PATH),
            "reference_output_path": str(REFERENCE_OUTPUT_PATH),
            "board_validation_report": board_report,
        },
        "storage": _disk_summary(WORKSPACE_DIR),
    }


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_upload() -> Path:
    file_storage = request.files.get("input_features")
    if file_storage is None or not file_storage.filename:
        raise ValueError("No .npy feature tensor was provided.")
    if not allowed_file(file_storage.filename):
        raise ValueError("Unsupported input type. Upload a NumPy .npy tensor only.")

    safe_name = secure_filename(file_storage.filename) or "input_features.npy"
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    file_storage.save(target)
    return target


@app.get("/")
def index() -> str:
    return render_template("rknn_encoder_index.html", diagnostics=collect_diagnostics())


@app.get("/api/status")
def api_status() -> Any:
    return jsonify(collect_diagnostics())


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"status": "ok", "npu": collect_diagnostics()["npu"]})


@app.post("/api/npu-self-test")
def api_npu_self_test() -> Any:
    try:
        report = run_encoder(
            input_path=REFERENCE_INPUT_PATH,
            output_dir=REPORT_DIR,
            reference_output=REFERENCE_OUTPUT_PATH,
        )
        return jsonify({"status": "ok", "diagnostics": collect_diagnostics(), "npu_report": report})
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "diagnostics": collect_diagnostics(),
        }), 400


@app.post("/api/run-encoder")
def api_run_encoder() -> Any:
    try:
        input_path = save_upload()
        run_dir = WORKSPACE_DIR / "outputs" / "poc_native_run"
        report = run_encoder(input_path=input_path, output_dir=run_dir)
        return jsonify({"status": "ok", "diagnostics": collect_diagnostics(), "npu_report": report})
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "diagnostics": collect_diagnostics(),
        }), 400


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
