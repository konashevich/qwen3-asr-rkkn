#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import torch
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from torch.onnx.errors import UnsupportedOperatorError
from transformers import AutoModel


TARGET_SAMPLE_RATE = 16000


class AudioTowerExportWrapper(torch.nn.Module):
    def __init__(self, audio_tower: torch.nn.Module):
        super().__init__()
        self.audio_tower = audio_tower

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        return self.audio_tower(input_features=input_features, feature_lens=feature_lens).last_hidden_state


class SingleChunkAudioTowerExportWrapper(torch.nn.Module):
    def __init__(self, audio_tower: torch.nn.Module, fixed_aftercnn_len: int):
        super().__init__()
        self.audio_tower = audio_tower
        self.fixed_aftercnn_len = int(fixed_aftercnn_len)

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        padded_feature = input_features.unsqueeze(0).unsqueeze(0)

        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d1(padded_feature))
        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d2(padded_embed))
        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d3(padded_embed))

        batch_size, channels, feature_bins, time_steps = padded_embed.size()
        padded_embed = self.audio_tower.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * feature_bins)
        )

        positional_embedding = (
            self.audio_tower.positional_embedding.positional_embedding[: self.fixed_aftercnn_len, :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        hidden_states = padded_embed[0, : self.fixed_aftercnn_len, :]
        cu_seqlens = torch.tensor(
            [0, self.fixed_aftercnn_len],
            dtype=torch.int32,
            device=input_features.device,
        )

        for encoder_layer in self.audio_tower.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        hidden_states = self.audio_tower.proj2(hidden_states)
        return hidden_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen3-ASR thinker.audio_tower to ONNX and validate it against PyTorch."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID or local path, for example Qwen/Qwen3-ASR-0.6B.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/encoder_onnx",
        help="Directory for the ONNX model, sample inputs, and metadata.",
    )
    parser.add_argument(
        "--onnx-name",
        default="qwen3_asr_encoder.onnx",
        help="Filename for the exported ONNX model.",
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help="Optional real audio file to derive export example features.",
    )
    parser.add_argument(
        "--synthetic-seconds",
        type=float,
        default=1.0,
        help="Length of synthetic audio to generate when --audio-file is not provided.",
    )
    parser.add_argument(
        "--synthetic-frequency",
        type=float,
        default=440.0,
        help="Frequency of the synthetic sine wave when --audio-file is not provided.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32"],
        default="float32",
        help="Model/export dtype. Float32 is the only supported export mode for now.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only read from locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX Runtime validation after export.",
    )
    return parser.parse_args()


def make_synthetic_waveform(seconds: float, frequency: float, sample_rate: int) -> np.ndarray:
    sample_count = max(int(round(seconds * sample_rate)), sample_rate // 10)
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    waveform = 0.1 * np.sin(2.0 * np.pi * frequency * timeline)
    return waveform.astype(np.float32)


def load_waveform(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.audio_file:
        path = Path(args.audio_file).expanduser().resolve()
        waveform, sample_rate = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
        metadata = {
            "audio_source": str(path),
            "audio_origin": "file",
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_seconds": float(len(waveform) / TARGET_SAMPLE_RATE),
        }
        return waveform.astype(np.float32), metadata

    waveform = make_synthetic_waveform(
        seconds=args.synthetic_seconds,
        frequency=args.synthetic_frequency,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    metadata = {
        "audio_source": None,
        "audio_origin": "synthetic",
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration_seconds": float(len(waveform) / TARGET_SAMPLE_RATE),
        "synthetic_frequency": args.synthetic_frequency,
    }
    return waveform, metadata


def build_example_inputs(
    model_id: str,
    waveform: np.ndarray,
    local_files_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    processor = Qwen3ASRProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        fix_mistral_regex=True,
    )
    batch = processor(
        text=[processor.audio_token],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
    )
    input_features = batch["input_features"][0].to(torch.float32).contiguous()
    feature_lens = batch["feature_attention_mask"].sum(-1).to(torch.long).contiguous()
    metadata = {
        "processor_class": type(processor).__name__,
        "feature_shape": list(input_features.shape),
        "feature_lens": feature_lens.tolist(),
        "audio_token": processor.audio_token,
    }
    return input_features, feature_lens, metadata


def load_audio_tower(model_id: str, local_files_only: bool) -> torch.nn.Module:
    model = AutoModel.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    model.eval()
    audio_tower = model.thinker.audio_tower
    audio_tower.config._attn_implementation = "eager"
    for layer in audio_tower.layers:
        layer.self_attn.config._attn_implementation = "eager"
    audio_tower.eval()
    return audio_tower


def export_onnx(
    wrapper: torch.nn.Module,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    onnx_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        wrapper,
        (input_features, feature_lens),
        str(onnx_path),
        input_names=["input_features", "feature_lens"],
        output_names=["last_hidden_state"],
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )


def run_pytorch_reference(
    wrapper: torch.nn.Module,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        output = wrapper(input_features, feature_lens)
    return output.detach().cpu().numpy()


def validate_onnx(
    onnx_path: Path,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    reference_output: np.ndarray,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    feed_dict = {"input_features": input_features.cpu().numpy()}
    if "feature_lens" in input_names:
        feed_dict["feature_lens"] = feature_lens.cpu().numpy()
    outputs = session.run(
        ["last_hidden_state"],
        feed_dict,
    )
    onnx_output = outputs[0]
    abs_diff = np.abs(reference_output - onnx_output)
    return {
        "onnx_input_names": input_names,
        "reference_shape": list(reference_output.shape),
        "onnx_shape": list(onnx_output.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


def collect_onnx_metadata(onnx_path: Path) -> dict[str, Any]:
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "ir_version": model.ir_version,
        "opset_imports": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(op_counts.items())),
    }


def save_numpy_inputs(output_dir: Path, input_features: torch.Tensor, feature_lens: torch.Tensor) -> str:
    npz_path = output_dir / "sample_inputs.npz"
    np.savez(
        npz_path,
        input_features=input_features.cpu().numpy(),
        feature_lens=feature_lens.cpu().numpy(),
    )
    return str(npz_path)


def write_metadata(output_dir: Path, payload: dict[str, Any]) -> str:
    metadata_path = output_dir / "export_metadata.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(metadata_path)


def attempt_export(
    wrapper: torch.nn.Module,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    onnx_path: Path,
    opset: int,
    skip_validation: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    reference_output = run_pytorch_reference(wrapper, input_features, feature_lens)
    export_onnx(wrapper, input_features, feature_lens, onnx_path, opset=opset)
    onnx_metadata = collect_onnx_metadata(onnx_path)
    validation = None
    if not skip_validation:
        validation = validate_onnx(onnx_path, input_features, feature_lens, reference_output)
    return onnx_metadata, validation, None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, audio_metadata = load_waveform(args)
    input_features, feature_lens, processor_metadata = build_example_inputs(
        args.model,
        waveform,
        local_files_only=args.local_files_only,
    )
    audio_tower = load_audio_tower(args.model, local_files_only=args.local_files_only)

    onnx_path = output_dir / args.onnx_name
    sample_inputs_path = save_numpy_inputs(output_dir, input_features, feature_lens)

    status = "ok"
    error = None
    onnx_metadata = None
    validation = None
    export_mode = "full_audio_tower"
    try:
        full_wrapper = AudioTowerExportWrapper(audio_tower)
        full_wrapper.eval()
        onnx_metadata, validation, _ = attempt_export(
            full_wrapper,
            input_features,
            feature_lens,
            onnx_path,
            opset=args.opset,
            skip_validation=args.skip_validation,
        )
    except UnsupportedOperatorError as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "hint": "The current audio_tower forward path uses pad_sequence-based chunk padding, which is not exportable in this ONNX pass.",
        }
        try:
            export_mode = "single_chunk_audio_tower_core"
            onnx_path = output_dir / "qwen3_asr_encoder_single_chunk.onnx"
            single_chunk_wrapper = SingleChunkAudioTowerExportWrapper(
                audio_tower,
                fixed_aftercnn_len=int(reference_output_shape(feature_lens)),
            )
            single_chunk_wrapper.eval()
            onnx_metadata, validation, _ = attempt_export(
                single_chunk_wrapper,
                input_features,
                feature_lens,
                onnx_path,
                opset=args.opset,
                skip_validation=args.skip_validation,
            )
            status = "ok"
            error = {
                "fallback_from": error,
                "message": "Full audio_tower export failed, but reduced single-chunk core export succeeded.",
            }
        except Exception as fallback_exc:  # pragma: no cover - failure path is still recorded in metadata
            status = "failed"
            error = {
                "primary": error,
                "fallback": {
                    "type": type(fallback_exc).__name__,
                    "message": str(fallback_exc),
                },
            }
    except Exception as exc:  # pragma: no cover - failure path is still recorded in metadata
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    metadata = {
        "status": status,
        "model": args.model,
        "export_target": "thinker.audio_tower",
        "export_mode": export_mode,
        "onnx_path": str(onnx_path),
        "sample_inputs": sample_inputs_path,
        "audio": audio_metadata,
        "processor": processor_metadata,
        "onnx": onnx_metadata,
        "validation": validation,
        "error": error,
        "notes": [
            "Static-shape export only in this first version.",
            "The exported entrypoint is the speech encoder only, not the full multimodal model.",
            "Example inputs are derived from Qwen3-ASR's own processor to keep feature formatting aligned with the model.",
        ],
    }
    metadata_path = write_metadata(output_dir, metadata)

    print(json.dumps({
        "status": status,
        "onnx_path": str(onnx_path),
        "metadata_path": metadata_path,
        "validation": validation,
        "error": error,
    }, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 1


def reference_output_shape(feature_lens: torch.Tensor) -> int:
    feature_len = int(feature_lens.to(torch.long).cpu().numpy().reshape(-1)[0])
    input_lengths_leave = feature_len % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feature_len // 100) * 13
    return int(output_lengths)


if __name__ == "__main__":
    raise SystemExit(main())