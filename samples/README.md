# Sample Set

This directory is for the fixed baseline validation set required by Phase 1.

Target coverage:

- 5 short Chinese clips
- 3 short English clips
- 2 longer clips
- 1 noisy clip

Suggested layout:

- `zh/`
- `en/`
- `long/`
- `noisy/`

Expected metadata file:

- `manifest.csv`

Use `manifest.csv` to record:

- relative audio path
- language hint if any
- clip category
- expected reference text
- notes

Once audio is present, run baseline validation with:

```bash
.venv-qwen3-asr/bin/python validate_baseline.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --audio samples \
  --output-dir outputs/baseline
```