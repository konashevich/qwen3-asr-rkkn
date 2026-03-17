# Qwen3-ASR Model Inspection

- Model type: qwen3_asr
- Audio token id: None
- Pad token id: None
- Supported languages declared in config: 30

## Phase 2 Split

- Python preprocessing: audio normalization, chunking, and prompt construction happen outside the model graph.
- Speech encoder: thinker.audio_tower (Qwen3ASRAudioEncoder)
- Decoder core: thinker.model (Qwen3ASRThinkerTextModel)
- Decoder head: thinker.lm_head (Linear)
- Multimodal bridge: audio encoder outputs are inserted into text embeddings at audio placeholder token positions

## Audio Encoder

- Mel bins: 128
- Hidden size: 896
- Output dim: 1024
- Encoder layers: 18
- Attention heads: 14
- FFN dim: 3584
- Downsample hidden size: 480
- Window sizes: n_window=50, n_window_infer=800
- Conv chunk size: 500
- Activation: gelu

## Decoder

- Hidden size: 1024
- Intermediate size: 3072
- Layers: 28
- Attention heads: 16
- Key/value heads: 8
- Vocab size: 151936
- Rope theta: 1000000

## Export Notes

- thinker.audio_tower is the first export target for ONNX and RKNN proof-of-concept
- audio features are scattered into text embeddings inside thinker.forward, so full native decoder mapping depends on reproducing this bridge
- baseline inference performs audio chunking outside the PyTorch graph, so CPU-side preprocessing is likely required even in a hybrid deployment
