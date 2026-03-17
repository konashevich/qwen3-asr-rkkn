#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
from transformers import AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Qwen3-ASR structure and write a Phase 2 component summary."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID or local path, for example Qwen/Qwen3-ASR-0.6B.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inspect",
        help="Directory for the generated JSON and Markdown reports.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only read model/config files that already exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--include-processor",
        action="store_true",
        help="Also load the processor metadata to inspect tokenizer and feature extractor types.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def extract_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def collect_component_summary(config: Qwen3ASRConfig) -> dict[str, Any]:
    thinker = config.thinker_config
    audio = thinker.audio_config
    text = thinker.text_config

    return {
        "root": {
            "model_type": extract_attr(config, "model_type"),
            "pad_token_id": extract_attr(config, "pad_token_id"),
            "audio_token_id": extract_attr(config, "audio_token_id"),
            "support_languages": extract_attr(config, "support_languages", []),
        },
        "pipeline_split": {
            "python_preprocessing": [
                "audio normalization and resampling in qwen_asr.inference.utils.normalize_audios",
                "audio chunking in qwen_asr.inference.utils.split_audio_into_chunks",
                "prompt construction via processor.apply_chat_template",
            ],
            "speech_encoder": {
                "path": "thinker.audio_tower",
                "class_name": "Qwen3ASRAudioEncoder",
                "main_input_name": "input_features",
                "submodules": [
                    "conv2d1",
                    "conv2d2",
                    "conv2d3",
                    "conv_out",
                    "layers[*]",
                    "ln_post",
                    "proj1",
                    "proj2",
                ],
            },
            "decoder": {
                "path": "thinker.model",
                "class_name": "Qwen3ASRThinkerTextModel",
                "submodules": [
                    "embed_tokens",
                    "layers[*]",
                    "norm",
                    "rotary_emb",
                ],
            },
            "decoder_head": {
                "path": "thinker.lm_head",
                "class_name": "Linear",
            },
            "multimodal_bridge": {
                "path": "thinker.forward",
                "behavior": "audio encoder outputs are inserted into text embeddings at audio placeholder token positions",
            },
        },
        "audio_encoder_config": {
            "num_mel_bins": extract_attr(audio, "num_mel_bins"),
            "d_model": extract_attr(audio, "d_model"),
            "output_dim": extract_attr(audio, "output_dim"),
            "encoder_layers": extract_attr(audio, "encoder_layers"),
            "encoder_attention_heads": extract_attr(audio, "encoder_attention_heads"),
            "encoder_ffn_dim": extract_attr(audio, "encoder_ffn_dim"),
            "downsample_hidden_size": extract_attr(audio, "downsample_hidden_size"),
            "n_window": extract_attr(audio, "n_window"),
            "n_window_infer": extract_attr(audio, "n_window_infer"),
            "conv_chunksize": extract_attr(audio, "conv_chunksize"),
            "activation_function": extract_attr(audio, "activation_function"),
        },
        "text_decoder_config": {
            "hidden_size": extract_attr(text, "hidden_size"),
            "intermediate_size": extract_attr(text, "intermediate_size"),
            "num_hidden_layers": extract_attr(text, "num_hidden_layers"),
            "num_attention_heads": extract_attr(text, "num_attention_heads"),
            "num_key_value_heads": extract_attr(text, "num_key_value_heads"),
            "vocab_size": extract_attr(text, "vocab_size"),
            "rope_theta": extract_attr(text, "rope_theta"),
        },
        "export_relevance": {
            "encoder_candidate": "thinker.audio_tower is the first export target for ONNX and RKNN proof-of-concept",
            "bridge_constraint": "audio features are scattered into text embeddings inside thinker.forward, so full native decoder mapping depends on reproducing this bridge",
            "preprocessing_constraint": "baseline inference performs audio chunking outside the PyTorch graph, so CPU-side preprocessing is likely required even in a hybrid deployment",
        },
    }


def render_markdown(summary: dict[str, Any], processor_info: dict[str, Any] | None) -> str:
    root = summary["root"]
    audio = summary["audio_encoder_config"]
    text = summary["text_decoder_config"]
    split = summary["pipeline_split"]

    lines = [
        "# Qwen3-ASR Model Inspection",
        "",
        f"- Model type: {root['model_type']}",
        f"- Audio token id: {root['audio_token_id']}",
        f"- Pad token id: {root['pad_token_id']}",
        f"- Supported languages declared in config: {len(root['support_languages'])}",
        "",
        "## Phase 2 Split",
        "",
        "- Python preprocessing: audio normalization, chunking, and prompt construction happen outside the model graph.",
        f"- Speech encoder: {split['speech_encoder']['path']} ({split['speech_encoder']['class_name']})",
        f"- Decoder core: {split['decoder']['path']} ({split['decoder']['class_name']})",
        f"- Decoder head: {split['decoder_head']['path']} ({split['decoder_head']['class_name']})",
        f"- Multimodal bridge: {split['multimodal_bridge']['behavior']}",
        "",
        "## Audio Encoder",
        "",
        f"- Mel bins: {audio['num_mel_bins']}",
        f"- Hidden size: {audio['d_model']}",
        f"- Output dim: {audio['output_dim']}",
        f"- Encoder layers: {audio['encoder_layers']}",
        f"- Attention heads: {audio['encoder_attention_heads']}",
        f"- FFN dim: {audio['encoder_ffn_dim']}",
        f"- Downsample hidden size: {audio['downsample_hidden_size']}",
        f"- Window sizes: n_window={audio['n_window']}, n_window_infer={audio['n_window_infer']}",
        f"- Conv chunk size: {audio['conv_chunksize']}",
        f"- Activation: {audio['activation_function']}",
        "",
        "## Decoder",
        "",
        f"- Hidden size: {text['hidden_size']}",
        f"- Intermediate size: {text['intermediate_size']}",
        f"- Layers: {text['num_hidden_layers']}",
        f"- Attention heads: {text['num_attention_heads']}",
        f"- Key/value heads: {text['num_key_value_heads']}",
        f"- Vocab size: {text['vocab_size']}",
        f"- Rope theta: {text['rope_theta']}",
        "",
        "## Export Notes",
        "",
        f"- {summary['export_relevance']['encoder_candidate']}",
        f"- {summary['export_relevance']['bridge_constraint']}",
        f"- {summary['export_relevance']['preprocessing_constraint']}",
    ]

    if processor_info is not None:
        lines.extend(
            [
                "",
                "## Processor",
                "",
                f"- Processor class: {processor_info['processor_class']}",
                f"- Tokenizer class: {processor_info['tokenizer_class']}",
                f"- Feature extractor class: {processor_info['feature_extractor_class']}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Qwen3ASRConfig.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    summary = collect_component_summary(config)

    processor_info = None
    if args.include_processor:
        processor = AutoProcessor.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            fix_mistral_regex=True,
        )
        processor_info = {
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(getattr(processor, "tokenizer", None)).__name__,
            "feature_extractor_class": type(getattr(processor, "feature_extractor", None)).__name__,
        }

    json_path = output_dir / "model_inspection.json"
    markdown_path = output_dir / "model_inspection.md"
    payload = {"summary": summary, "processor": processor_info}

    json_path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary, processor_info), encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())