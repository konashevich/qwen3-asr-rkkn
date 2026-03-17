#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from qwen_asr import Qwen3ASRModel
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HF_HOME = Path(
    os.environ.get("QWEN3_ASR_HF_HOME", "/mnt/merged_ssd/qwen3-asr-model-cache/huggingface")
)
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))

MODEL_ID = os.environ.get("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
MAX_NEW_TOKENS = int(os.environ.get("QWEN3_ASR_MAX_NEW_TOKENS", "512"))
MAX_BATCH_SIZE = int(os.environ.get("QWEN3_ASR_MAX_BATCH_SIZE", "1"))
HOST = os.environ.get("QWEN3_ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("QWEN3_ASR_PORT", "7861"))
ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".aac",
    ".webm",
    ".mp4",
}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def resolve_model_reference() -> str:
    if os.environ.get("QWEN3_ASR_MODEL"):
        return MODEL_ID

    snapshot_root = DEFAULT_HF_HOME / "hub" / "models--Qwen--Qwen3-ASR-0.6B" / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return MODEL_ID


MODEL_REFERENCE = resolve_model_reference()


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


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Qwen3ASRModel | None = None
        self._load_seconds: float | None = None

    def status(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_reference": MODEL_REFERENCE,
            "loaded": self._model is not None,
            "load_seconds": self._load_seconds,
            "cache_home": str(DEFAULT_HF_HOME),
            "max_batch_size": MAX_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

    def ensure_loaded(self) -> Qwen3ASRModel:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                started = time.perf_counter()
                self._model = Qwen3ASRModel.from_pretrained(
                    MODEL_REFERENCE,
                    max_inference_batch_size=MAX_BATCH_SIZE,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                self._load_seconds = time.perf_counter() - started
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        context: str,
        return_time_stamps: bool,
    ) -> dict[str, Any]:
        model = self.ensure_loaded()
        started = time.perf_counter()
        items = model.transcribe(
            str(audio_path),
            language=language or None,
            context=context,
            return_time_stamps=return_time_stamps,
        )
        elapsed = time.perf_counter() - started
        serialized = [_jsonable(item) for item in items]
        parsed_text = "\n".join(item.get("text", "") for item in serialized).strip()
        return {
            "audio_path": str(audio_path),
            "elapsed_seconds": elapsed,
            "result_count": len(serialized),
            "raw_output": serialized,
            "parsed_text": parsed_text,
        }


runtime = ModelRuntime()


def collect_diagnostics() -> dict[str, Any]:
    board_model = _read_text(Path("/proc/device-tree/model"))
    compatible_text = _read_text(Path("/proc/device-tree/compatible")) or ""
    compatible = [item for item in compatible_text.replace("\x00", "\n").splitlines() if item]
    encoder_report = _read_json(WORKSPACE_DIR / "outputs/device_run_on_board/validation_report.json")
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
            "rknnlite_site_present": Path(
                "/home/pi/npu_env/lib/python3.10/site-packages/rknnlite"
            ).exists(),
            "encoder_rknn_path": str(
                WORKSPACE_DIR / "outputs/rknn/qwen3_asr_encoder_single_chunk_rk3588.rknn"
            ),
            "board_validation_report": encoder_report,
        },
        "runtime": runtime.status(),
        "storage": _disk_summary(DEFAULT_HF_HOME.parent),
    }


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_upload() -> Path:
    file_storage = request.files.get("audio")
    if file_storage is None or not file_storage.filename:
        raise ValueError("No audio file was provided.")
    if not allowed_file(file_storage.filename):
        raise ValueError(
            "Unsupported audio type. Use one of: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )

    safe_name = secure_filename(file_storage.filename) or "audio_upload"
    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    file_storage.save(target)
    return target


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        diagnostics=collect_diagnostics(),
        allowed_extensions=", ".join(sorted(ALLOWED_EXTENSIONS)),
    )


@app.get("/api/status")
def api_status() -> Any:
    return jsonify(collect_diagnostics())


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"status": "ok", "runtime": runtime.status()})


@app.post("/api/transcribe")
def api_transcribe() -> Any:
    audio_path: Path | None = None
    try:
        audio_path = save_upload()
        result = runtime.transcribe(
            audio_path=audio_path,
            language=(request.form.get("language") or "").strip() or None,
            context=(request.form.get("context") or "").strip(),
            return_time_stamps=(request.form.get("return_time_stamps") == "true"),
        )
        return jsonify({
            "status": "ok",
            "diagnostics": collect_diagnostics(),
            "transcription": result,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "diagnostics": collect_diagnostics(),
        }), 400


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)