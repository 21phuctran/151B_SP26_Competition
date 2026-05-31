
"""Single-entry-point inference for CSE 151B SP26 competition submission.

Loads Qwen3-4B-Thinking-2507 (bf16) + LoRA adapter from HuggingFace Hub,
generates responses for every problem in `private.jsonl`, and writes a
properly-escaped `submission.csv` with columns (id, response).

Usage:
    python run_inference.py                       # full run, writes submission.csv
    python run_inference.py --limit 20            # smoke test on first 20 problems
    python run_inference.py --no-lora             # baseline (no adapter) for ablation
    python run_inference.py --private-path foo.jsonl --output foo.csv

The single `run_inference()` function is the competition-required entry point —
it returns nothing and produces the CSV as a side effect, mirroring how the
TA's verification harness will call it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# These imports are deferred inside main() so `python run_inference.py --help`
# works without GPU/vLLM installed. Keep that at the top of the file for the
# helper module (pure stdlib + re) which IS safe to import at module scope.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_helpers import (
    build_prompt,
    question_type,
    pick_representative_response,
)


# ─── Configuration (final hyperparameters used for the submission) ────────────

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
HF_ADAPTER  = "tsinha/qwen3-4b-thinking-cse151b"  # update to your actual repo
MAX_MODEL_LEN = 16384
PRIVATE_CACHE = "cache/private_samples.jsonl"   # per-bucket append, crash-safe

# Per-bucket max_tokens. MCQ failures were ~70% truncation (model never wrote
# \boxed{X} before hitting 8192 on hard problems), so we give it more room. FF
# stays at 8192 — failures there were format issues, not budget overflow.
MAX_TOKENS_MCQ = 12288
MAX_TOKENS_FF  = 8192

# Sampling: matches the notebook's Section 7 final config.
# MCQ: low temp, few samples (vote wastes budget on letters).
# FF:  higher temp, more samples (self-consistency on numeric/symbolic answers).
SAMPLING_MCQ = dict(n=3,  temperature=0.3, top_p=0.95, top_k=20, min_p=0.0)
SAMPLING_FF  = dict(n=8, temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)


# ─── Core pipeline ────────────────────────────────────────────────────────────

def run_inference(
    private_path: str = "data/private.jsonl",
    output_path: str = "submission.csv",
    *,
    use_lora: bool = True,
    limit: int | None = None,
) -> None:
    """Load the model, run inference on `private_path`, write CSV to `output_path`.

    This is the single function the TA will call. Self-contained: loads the
    model, generates responses, votes, and writes the CSV in one call.
    """
    import os
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from judger import Judger

    # ── 1. Load problems ──────────────────────────────────────────────────────
    problems = [json.loads(line) for line in open(private_path)]
    if limit is not None:
        problems = problems[:limit]
    print(f"Loaded {len(problems)} problems from {private_path}")

    # ── 1b. Load any cached samples from a previous (interrupted) run ─────────
    # Resilience: if a previous invocation crashed mid-FF-multi, we don't
    # want to redo MCQ + FF-single. The cache is keyed by problem id.
    os.makedirs(os.path.dirname(PRIVATE_CACHE), exist_ok=True)
    cached_samples: dict = {}
    if os.path.exists(PRIVATE_CACHE):
        with open(PRIVATE_CACHE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cached_samples[r["id"]] = r["samples"]
                except Exception:
                    pass   # tolerate a partial last line from a prior crash
        print(f"  Resuming: {len(cached_samples)} questions already cached "
              f"at {PRIVATE_CACHE}")
    else:
        print(f"  Starting fresh (no cache yet)")

    # ── 2. Load tokenizer + model ─────────────────────────────────────────────
    print(f"Loading tokenizer + base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        # 0.92 on Blackwell (96GB) — ~30% higher concurrent batch size vs 0.85.
        # Confirmed working in practice. On 24GB GPUs (A5000/A30) drop back
        # to 0.85.
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enforce_eager=False,
    )
    lora_request = None
    if use_lora:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16, max_loras=1)
        lora_request = LoRARequest("qlora_rft", 1, HF_ADAPTER)
        print(f"  LoRA adapter: {HF_ADAPTER}")
    else:
        print("  LoRA adapter: DISABLED (baseline mode)")

    llm = LLM(**llm_kwargs)
    judger = Judger(strict_extract=False)
    print("Model loaded.")

    # ── 3. Bucket problems by type (skip questions already cached) ────────────
    buckets = {"mcq": [], "ff_single": [], "ff_multi": []}
    samples_per_idx: list[list[str] | None] = [None] * len(problems)
    idx_to_item: dict[int, dict] = {}
    for idx, item in enumerate(problems):
        idx_to_item[idx] = item
        if item["id"] in cached_samples:
            samples_per_idx[idx] = cached_samples[item["id"]]
            continue
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        buckets[question_type(item)].append((idx, prompt_text))

    n_cached = sum(1 for s in samples_per_idx if s is not None)
    n_remaining = sum(len(b) for b in buckets.values())
    print(f"  Cached this run: {n_cached}  |  To generate: {n_remaining}")
    print(f"Buckets — MCQ: {len(buckets['mcq'])}, "
          f"FF-single: {len(buckets['ff_single'])}, "
          f"FF-multi: {len(buckets['ff_multi'])}")

    # ── 4. Generate per bucket (append to cache after each bucket) ────────────

    def run_bucket(name: str, params: SamplingParams) -> None:
        bucket = buckets[name]
        if not bucket:
            print(f"\nSkipping {name}: nothing to generate (all cached).")
            return
        idxs, prompts = zip(*bucket)
        print(f"\nGenerating {name}: {len(prompts)} prompts × N={params.n}")
        t0 = time.perf_counter()
        outputs = llm.generate(
            list(prompts),
            sampling_params=params,
            lora_request=lora_request,
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        # Fill in memory + append to cache so a later crash doesn't lose this bucket.
        new_records = []
        for idx, out in zip(idxs, outputs):
            s = [c.text.strip() for c in out.outputs]
            samples_per_idx[idx] = s
            new_records.append({"id": idx_to_item[idx]["id"], "samples": s})

        with open(PRIVATE_CACHE, "a") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")
        print(f"  appended {len(new_records)} → {PRIVATE_CACHE}")

    sampling_mcq = SamplingParams(max_tokens=MAX_TOKENS_MCQ, **SAMPLING_MCQ)
    sampling_ff  = SamplingParams(max_tokens=MAX_TOKENS_FF,  **SAMPLING_FF)

    run_bucket("mcq",       sampling_mcq)
    run_bucket("ff_single", sampling_ff)
    run_bucket("ff_multi",  sampling_ff)

    # ── 5. Pick representative response per problem + write CSV ───────────────
    print(f"\nWriting {output_path}")
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    with open(output_p, "w", newline="") as f:
        # QUOTE_ALL ensures commas / newlines / quotes inside the response are
        # escaped properly. Required because reasoning traces contain all three.
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])

        for idx, item in enumerate(problems):
            samples = samples_per_idx[idx] or []
            qtype = question_type(item)
            response = pick_representative_response(samples, qtype, judger=judger)
            if not response:
                n_empty += 1
                # Always emit a row so every id has coverage. The judger will
                # mark it wrong, but a missing id is worse (could fail validation).
                response = "(no response generated)"
            writer.writerow([item["id"], response])

    print(f"Wrote {len(problems)} rows to {output_path}")
    if n_empty:
        print(f"  WARNING: {n_empty} problems produced no usable response")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-path", default="data/private.jsonl",
                   help="Path to private.jsonl")
    p.add_argument("--output", default="submission.csv",
                   help="Output CSV path")
    p.add_argument("--no-lora", action="store_true",
                   help="Disable LoRA adapter (baseline ablation)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N problems (smoke test)")
    args = p.parse_args()

    run_inference(
        private_path=args.private_path,
        output_path=args.output,
        use_lora=not args.no_lora,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()