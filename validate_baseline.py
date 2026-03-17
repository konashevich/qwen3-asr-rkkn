#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from qwen_asr import Qwen3ASRModel


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline Qwen3-ASR inference and save gold-reference outputs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID or local path, for example Qwen/Qwen3-ASR-0.6B.",
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="One or more audio files or directories containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline",
        help="Directory for JSON, text, and timing outputs.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional forced language hint passed to Qwen3-ASR.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional decoding context prompt.",
    )
    parser.add_argument(
        "--return-time-stamps",
        action="store_true",
        help="Request timestamp output from the model.",
    )
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=1,
        help="Batch size passed to from_pretrained for baseline runs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens passed to from_pretrained.",
    )
    return parser.parse_args()


def collect_audio_files(entries: list[str]) -> list[Path]:
    audio_files: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio path does not exist: {path}")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
                    audio_files.append(candidate.resolve())
            continue
        raise ValueError(f"Unsupported audio input: {path}")

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for audio_file in audio_files:
        if audio_file not in seen:
            seen.add(audio_file)
            unique_files.append(audio_file)
    return unique_files


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> int:
    args = parse_args()
    audio_files = collect_audio_files(args.audio)
    if not audio_files:
        raise ValueError("No audio files were found in the provided --audio paths.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_load = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    load_seconds = time.perf_counter() - start_load

    summary: dict[str, Any] = {
        "model": args.model,
        "audio_count": len(audio_files),
        "language": args.language,
        "context": args.context,
        "return_time_stamps": args.return_time_stamps,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "supported_languages": model.get_supported_languages(),
        "samples": [],
    }

    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for audio_file in audio_files:
            started = time.perf_counter()
            transcriptions = model.transcribe(
                str(audio_file),
                context=args.context,
                language=args.language,
                return_time_stamps=args.return_time_stamps,
            )
            elapsed = time.perf_counter() - started

            serialized = [to_jsonable(item) for item in transcriptions]
            parsed_text = "\n".join(item.get("text", "") for item in serialized).strip()
            sample = {
                "audio_path": str(audio_file),
                "elapsed_seconds": elapsed,
                "result_count": len(serialized),
                "raw_output": serialized,
                "parsed_text": parsed_text,
            }
            summary["samples"].append(sample)
            jsonl_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

            text_path = output_dir / f"{safe_name(audio_file)}.txt"
            text_path.write_text(parsed_text + "\n", encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "results_jsonl": str(jsonl_path),
        "audio_count": len(audio_files),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"validate_baseline.py failed: {exc}", file=sys.stderr)
        raise