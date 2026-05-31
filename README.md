# CSE 151B SP26 Competition Submission

Team: PTS(D) — Tanya Sinha, Jiayi Gao, Phuc Tran
Private leaderboard score: **0.706**

## TL;DR for verification (200-question re-run)

```bash
# 1. Environment (Python 3.13, CUDA 12.8+ or 13.0)
python3.13 -m venv .venv
source .venv/bin/activate
pip install vllm==0.20.1 transformers torch

# 2. Run inference on a sampled subset
python run_inference.py \
    --no-lora \
    --private-path PATH/TO/your_200_sample.jsonl \
    --output submission.csv
```

That writes `submission.csv` with `(id, response)` columns. **No model weights to set up manually** — the base model is pulled from HF on first run. **No LoRA adapter** — submission uses the raw base model (`--no-lora`).

**Expected runtime for 200 questions:** ~5 hrs on a single 24 GB GPU (RTX PRO 6000 / L40S / A100). The MCQ bucket finishes in ~30 min; free-form buckets are the bottleneck (N=8 self-consistency with 8K-token thinking traces).

**Resumable:** the script appends per-bucket samples to `cache/private_samples.jsonl`. If a run is interrupted, just re-run the same command — completed buckets are skipped. Delete `cache/private_samples.jsonl` for a fresh start.

**Sanity smoke test (5 questions, ~5 min):**
```bash
python run_inference.py --no-lora --private-path PATH/TO/your_200_sample.jsonl --output /tmp/smoke.csv --limit 5
```
Should write 6 lines (header + 5) to `/tmp/smoke.csv`.

## Approach

Base model only — no fine-tuning. `Qwen/Qwen3-4B-Thinking-2507` loaded in bf16 via vLLM. Per-bucket self-consistency voting (N=3 for MCQ, N=8 for free-form) plus prompt engineering targeting decimal precision (≥10 sig figs), explicit blank counting for multi-`[ANS]` questions, and reliable `\boxed{}` extraction.

## Hardware & inference time (full 943-question private set)

| Item | Value |
|---|---|
| GPU used | NVIDIA RTX PRO 6000 Blackwell (24 GB) |
| Precision | bfloat16 |
| Engine | vLLM 0.20.1, `max_model_len=16384`, `gpu_memory_utilization=0.92` |
| Full 943-question generation time | ~25 hrs total |

## How to call `run_inference()`

Python API:
```python
from run_inference import run_inference

run_inference(
    private_path="private.jsonl",
    output_path="submission.csv",
    use_lora=False,   # submission uses base model only
    limit=None,       # or N for the first N questions
)
```

CLI:
```bash
python run_inference.py --no-lora \
    --private-path private.jsonl \
    --output submission.csv
```

The TA can also pass their own sampled subset via `--private-path PATH/TO/sample.jsonl`. All post-processing (per-bucket sampling configs, self-consistency voting, decimal precision handling, multi-blank handling, CSV escaping) is packed inside this one function. Calling it produces the final CSV — nothing else needed.

## Hyperparameters (final, baked into the code)

These are the values used to produce the submission; see [`run_inference.py`](run_inference.py):

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Thinking-2507` |
| `max_model_len` | 16384 |
| MCQ: `max_tokens` | 12288 |
| MCQ: sampling | `n=3, temperature=0.3, top_p=0.95, top_k=20` |
| Free-form: `max_tokens` | 8192 |
| Free-form: sampling | `n=8, temperature=0.7, top_p=0.95, top_k=20` |
| Voting | Whole-answer plurality over normalized extracted answers (see [`inference_helpers.py`](inference_helpers.py)) |

## Files

| File | Purpose |
|---|---|
| `run_inference.py` | Single-entry-point inference pipeline |
| `inference_helpers.py` | Prompts, voting, response selection |
| `judger.py` | Answer extraction + symbolic equivalence checking |
| `utils.py` | Helpers used by `judger.py` |
| `private.jsonl` | Private test set (default `--private-path`) |
| `data/public.jsonl` | Public set with gold answers |

The `cache/` directory is auto-created on first run for per-bucket sample caching (resumable).
 Note: `train.py` is exploratory code for QLoRA fine-tuning that was not used in the final submission.
