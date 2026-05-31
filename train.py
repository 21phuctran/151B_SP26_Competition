"""End-to-end training pipeline for the CSE 151B competition.

Four phases, each tolerant of resumption:
  1. Baseline rollouts on data/public.jsonl with the BASE model, no LoRA →
     cache/baseline_samples.jsonl  (per-bucket append, crash-safe)
  2. Rejection-sampling: pick questions where >=1 of the N samples is correct,
     build a chat-format training set, hold out 10% for later evaluation.
  3. QLoRA fine-tune on that set → ./qlora_adapter
  4. Eval: run inference (bf16 + LoRA, same as run_inference.py) on the
     held-out 10%, score against gold via judger.auto_judge, print accuracy
     by bucket. Run as a separate command (fresh process avoids GPU memory
     fragmentation from Phase 3's bnb model).

Designed to be run non-interactively, e.g.:
    # Full Phase 1 + 2 + 3 on the WHOLE public set (~12 hrs on A5000)
    nohup python train.py > train.log 2>&1 &

    # Or, smaller cache for faster iteration (~4 hrs)
    nohup python train.py --limit 400 > train.log 2>&1 &

    # Phase 4 eval with LoRA after training
    nohup python train.py --eval-only > eval.log 2>&1 &

    # Smoke test on holdout with BASE model + new prompts (no LoRA needed)
    nohup python train.py --eval-only --no-lora > smoke.log 2>&1 &

Skip-phase flags let you resume across pod restarts:
    --skip-baseline   reuse cache/baseline_samples.jsonl (regenerate if prompts changed!)
    --skip-train      stop after building train_dataset (no QLoRA pass)
    --eval-only       just run phase 4 (assumes ./qlora_adapter exists, or use --no-lora)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_helpers import build_prompt, question_type


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID            = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH   = "data/public.jsonl"
BASELINE_CACHE      = "cache/baseline_samples.jsonl"
HOLDOUT_SAMPLES     = "cache/holdout_samples.jsonl"     # phase_eval output, for failure analysis
HOLDOUT_IDS_PATH    = "cache/holdout_ids.json"
ADAPTER_PATH        = "./qlora_adapter"

MAX_TOKENS_MCQ      = 12288   # MCQ truncates often on hard problems; give more room
MAX_TOKENS_FF       = 8192    # FF rarely truncates; budget overflow was not the issue
MAX_MODEL_LEN       = 16384

# Baseline rollout sampling — matches notebook cell 23 final config.
SAMPLING_MCQ = dict(n=3,  temperature=0.3, top_p=0.95, top_k=20, min_p=0.0)
SAMPLING_FF  = dict(n=16, temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)

# Rejection-sampling / training-set construction.
HOLDOUT_FRAC          = 0.10
MIN_TRAIN_EXAMPLES    = 50
MAX_TRAIN_LEN_CHARS   = 14000


# ─── Phase 1 — baseline rollouts ──────────────────────────────────────────────

def phase_baseline(data_path: str, limit: int | None) -> None:
    """Generate baseline rollouts → cache/baseline_samples.jsonl.

    Uses bf16 (not bnb 4-bit) for speed — we tear vLLM down before training,
    so the GPU-sharing constraint that motivated bnb in the notebook doesn't
    apply here. Output distribution stays close to what run_inference.py
    produces at submission time.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = [json.loads(line) for line in open(data_path)]
    if limit is not None:
        data = data[:limit]
    print(f"[Phase 1] Loaded {len(data)} questions from {data_path}")

    os.makedirs(os.path.dirname(BASELINE_CACHE), exist_ok=True)
    cached_samples: dict = {}
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    cached_samples[r["id"]] = r["samples"]
                except Exception:
                    pass
        print(f"[Phase 1] Resuming: {len(cached_samples)} questions already cached")
    else:
        print(f"[Phase 1] Starting fresh (no cache yet)")

    print(f"[Phase 1] Loading tokenizer + base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=False,
    )
    print("[Phase 1] Model loaded.")

    buckets: dict[str, list] = {"mcq": [], "ff_single": [], "ff_multi": []}
    idx_to_item: dict[int, dict] = {}
    samples_per_idx: list = [None] * len(data)
    for idx, item in enumerate(data):
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
    print(f"[Phase 1] Cached this run: {n_cached}  |  To generate: {n_remaining}")
    print(f"[Phase 1] Buckets — MCQ: {len(buckets['mcq'])}, "
          f"FF-single: {len(buckets['ff_single'])}, FF-multi: {len(buckets['ff_multi'])}")

    sampling_mcq = SamplingParams(max_tokens=MAX_TOKENS_MCQ, **SAMPLING_MCQ)
    sampling_ff  = SamplingParams(max_tokens=MAX_TOKENS_FF,  **SAMPLING_FF)

    def run_bucket(name: str, params: SamplingParams) -> None:
        bucket = buckets[name]
        if not bucket:
            print(f"[Phase 1] Skipping {name}: nothing to generate (all cached).")
            return
        idxs, prompts = zip(*bucket)
        print(f"[Phase 1] Generating {name}: {len(prompts)} prompts × N={params.n}")
        t0 = time.perf_counter()
        outputs = llm.generate(list(prompts), sampling_params=params)
        print(f"[Phase 1]   done in {time.perf_counter() - t0:.1f}s")

        new_records = []
        for idx, out in zip(idxs, outputs):
            s = [c.text.strip() for c in out.outputs]
            samples_per_idx[idx] = s
            new_records.append({"id": idx_to_item[idx]["id"], "samples": s})

        with open(BASELINE_CACHE, "a") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")
        print(f"[Phase 1]   appended {len(new_records)} → {BASELINE_CACHE}")

    run_bucket("mcq",       sampling_mcq)
    run_bucket("ff_single", sampling_ff)
    run_bucket("ff_multi",  sampling_ff)

    # Free vLLM before training. Without this, QLoRA model load OOMs.
    print("[Phase 1] Tearing down vLLM to free GPU before training...")
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[Phase 1] GPU free after cleanup: "
              f"{torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    except Exception as e:
        print(f"[Phase 1] (torch cleanup skipped: {e})")


# ─── Phase 2 — build training set from baseline rollouts ──────────────────────

def phase_build_dataset(data_path: str):
    """Rejection-sample the baseline cache into a chat-format Dataset."""
    from datasets import Dataset
    sys.path.insert(0, ".")
    from judger import Judger

    judger = Judger(strict_extract=False)
    data = [json.loads(line) for line in open(data_path)]
    by_id = {item["id"]: item for item in data}

    if not Path(BASELINE_CACHE).exists():
        raise RuntimeError(
            f"No baseline cache at {BASELINE_CACHE}. Run phase 1 first "
            f"(remove --skip-baseline)."
        )

    cached = {json.loads(line)["id"]: json.loads(line)
              for line in open(BASELINE_CACHE)}
    print(f"[Phase 2] Loaded {len(cached)} cached rollouts from {BASELINE_CACHE}")

    rng = random.Random(42)
    all_ids = sorted(cached.keys())
    rng.shuffle(all_ids)
    n_holdout = max(1, int(len(all_ids) * HOLDOUT_FRAC))
    holdout_ids = set(all_ids[:n_holdout])
    train_ids   = set(all_ids[n_holdout:])
    print(f"[Phase 2] Hold-out: {len(holdout_ids)}  |  Train pool: {len(train_ids)}")

    def pick_best_sample(item, samples):
        gold = item["answer"]
        gold_list = gold if isinstance(gold, list) else [gold]
        is_mcq = bool(item.get("options"))
        for s in samples:
            if is_mcq:
                m = re.search(r"\\boxed\{\s*\(?\s*([A-Za-z])", s.split("</think>")[-1])
                if m and m.group(1).upper() == str(gold).strip().upper():
                    return s
                continue
            try:
                if judger.auto_judge(pred=s, gold=gold_list,
                                     options=[[]] * len(gold_list)):
                    return s
            except Exception:
                pass
        return None

    training_examples = []
    n_no_correct = 0
    n_truncated  = 0
    for qid in train_ids:
        item = by_id.get(qid)
        if item is None:
            continue
        samples = cached[qid]["samples"]
        chosen = pick_best_sample(item, samples)
        if chosen is None:
            n_no_correct += 1
            continue
        if len(chosen) > MAX_TRAIN_LEN_CHARS:
            n_truncated += 1
            continue
        system, user = build_prompt(item["question"], item.get("options"))
        training_examples.append({
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": chosen},
            ],
            "id": qid,
        })

    random.Random(0).shuffle(training_examples)
    print(f"[Phase 2] Training examples: {len(training_examples)}")
    print(f"[Phase 2]   (skipped {n_no_correct} no-correct, {n_truncated} oversized)")

    if len(training_examples) < MIN_TRAIN_EXAMPLES:
        raise RuntimeError(
            f"Only {len(training_examples)} correct rollouts — too few to fine-tune. "
            f"Increase --limit and re-run phase 1."
        )

    train_dataset = Dataset.from_list(
        [{"messages": ex["messages"]} for ex in training_examples]
    )

    os.makedirs("cache", exist_ok=True)
    with open(HOLDOUT_IDS_PATH, "w") as f:
        json.dump(sorted(holdout_ids), f)
    print(f"[Phase 2] Held-out ids saved → {HOLDOUT_IDS_PATH}")
    return train_dataset


# ─── Phase 3 — QLoRA fine-tune ────────────────────────────────────────────────

def phase_train(train_dataset) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print("[Phase 3] Loading tokenizer + 4-bit base model for QLoRA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    qlora_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    qlora_model = prepare_model_for_kbit_training(qlora_model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    qlora_model = get_peft_model(qlora_model, lora_config)
    qlora_model.config.use_cache = False
    qlora_model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="./qlora_checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,                 # was 2; more epochs on cleaner data
        learning_rate=5e-5,
        bf16=True,
        max_length=4096,
        # assistant_only_loss requires {% generation %} markers in the chat template;
        # Qwen3-Thinking's template doesn't have them and TRL refuses to patch it.
        # Training on all tokens (system+user+assistant) is fine — the system prompt
        # is fixed so the model just learns to reproduce a constant prefix, which
        # doesn't hurt generation. Wastes some loss budget but it's the safe path.
        assistant_only_loss=False,
        logging_steps=10,
        save_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=qlora_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("[Phase 3] Starting QLoRA training...")
    trainer.train()
    print("[Phase 3] QLoRA training complete.")

    qlora_model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"[Phase 3] LoRA adapter saved → {ADAPTER_PATH}")


# ─── Baseline eval — score cache/baseline_samples.jsonl against gold ──────────

def phase_eval_baseline(data_path: str, holdout_only: bool = False) -> None:
    """Score the BASE-model rollouts already in cache/baseline_samples.jsonl.

    No GPU needed — uses the cached samples + same voting + judging logic as
    Phase 4. Tells you the base-model accuracy on whichever questions you've
    already rolled out, so you can quote a clean delta after QLoRA training.

    If holdout_only=True, restrict scoring to the same 40 holdout ids that
    Phase 4 used — giving an apples-to-apples base-vs-LoRA comparison.
    """
    sys.path.insert(0, ".")
    from judger import Judger
    from inference_helpers import (
        pick_representative_response,
        extract_letter_mcq,
    )

    if not Path(BASELINE_CACHE).exists():
        raise RuntimeError(f"No baseline cache at {BASELINE_CACHE}. Run phase 1 first.")

    cached = {json.loads(line)["id"]: json.loads(line)["samples"]
              for line in open(BASELINE_CACHE)}
    data = [json.loads(line) for line in open(data_path)]
    items = [item for item in data if item["id"] in cached]
    if holdout_only:
        if not Path(HOLDOUT_IDS_PATH).exists():
            raise RuntimeError(f"No holdout ids at {HOLDOUT_IDS_PATH}.")
        holdout_ids = set(json.load(open(HOLDOUT_IDS_PATH)))
        items = [item for item in items if item["id"] in holdout_ids]
        print(f"[Baseline] Scoring {len(items)} holdout cached rollouts "
              f"(apples-to-apples vs Phase 4)")
    else:
        print(f"[Baseline] Scoring {len(items)} cached rollouts")

    judger = Judger(strict_extract=False)
    counts = {"mcq": [0, 0], "ff_single": [0, 0], "ff_multi": [0, 0]}

    for item in items:
        qtype = question_type(item)
        counts[qtype][1] += 1
        samples = cached[item["id"]]
        if not samples:
            continue
        gold = item["answer"]
        picked = pick_representative_response(samples, qtype, judger=judger)

        if qtype == "mcq":
            picked_letter = extract_letter_mcq(picked)
            correct = picked_letter == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            opts = item.get("options") or []
            options_arg = [opts if opts else []] * len(gold_list)
            try:
                correct = judger.auto_judge(
                    pred=picked, gold=gold_list, options=options_arg)
            except Exception:
                correct = False

        if correct:
            counts[qtype][0] += 1

    print("\n[Baseline] ────── BASE-MODEL ACCURACY (cached rollouts) ──────")
    total_c, total_n = 0, 0
    for qtype in ("mcq", "ff_single", "ff_multi"):
        c, n = counts[qtype]
        total_c += c
        total_n += n
        pct = (100 * c / n) if n else 0
        print(f"[Baseline]   {qtype:10s}: {c:3d}/{n:3d}  ({pct:5.1f}%)")
    pct = (100 * total_c / total_n) if total_n else 0
    print(f"[Baseline]   {'overall':10s}: {total_c:3d}/{total_n:3d}  ({pct:5.1f}%)")


# ─── Phase 4 — eval on the 10% held-out portion of public.jsonl ───────────────

def phase_eval(data_path: str, use_lora: bool = True) -> None:
    """Run inference on the held-out 10% and score against gold.

    Mirrors run_inference.py's engine config and inference_helpers voting, so
    accuracy here approximates what you'd get at submission time.

    With use_lora=True (default): bf16 + LoRA adapter, requires ./qlora_adapter.
    With use_lora=False: base model only. Useful as a SMOKE TEST after editing
    system prompts — tells you whether the new prompts beat the cached baseline
    of 62.5% before committing to a full Phase 1 regeneration.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    sys.path.insert(0, ".")
    from judger import Judger
    from inference_helpers import (
        pick_representative_response,
        extract_letter_mcq,
    )

    if not Path(HOLDOUT_IDS_PATH).exists():
        raise RuntimeError(
            f"No holdout ids at {HOLDOUT_IDS_PATH}. Run phase 2 first."
        )
    if use_lora and not Path(ADAPTER_PATH).exists():
        raise RuntimeError(
            f"No LoRA adapter at {ADAPTER_PATH}. Run phase 3 first "
            f"(or pass --no-lora for a base-model smoke test)."
        )

    holdout_ids = set(json.load(open(HOLDOUT_IDS_PATH)))
    data = [json.loads(line) for line in open(data_path)]
    holdout = [item for item in data if item["id"] in holdout_ids]
    mode_label = "BASE-only smoke test" if not use_lora else "BASE + LoRA"
    print(f"[Phase 4] Evaluating on {len(holdout)} held-out items ({mode_label})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        enforce_eager=False,
    )
    if use_lora:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16, max_loras=1)
    llm = LLM(**llm_kwargs)
    lora_request = (LoRARequest("qlora_rft", 1, ADAPTER_PATH)
                    if use_lora else None)
    judger = Judger(strict_extract=False)
    print(f"[Phase 4] Model loaded ({mode_label}).")

    buckets: dict[str, list] = {"mcq": [], "ff_single": [], "ff_multi": []}
    for idx, item in enumerate(holdout):
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        buckets[question_type(item)].append((idx, prompt_text))
    print(f"[Phase 4] Buckets — MCQ: {len(buckets['mcq'])}, "
          f"FF-single: {len(buckets['ff_single'])}, FF-multi: {len(buckets['ff_multi'])}")

    samples_per_idx: list = [None] * len(holdout)
    sampling_mcq = SamplingParams(max_tokens=MAX_TOKENS_MCQ, **SAMPLING_MCQ)
    sampling_ff  = SamplingParams(max_tokens=MAX_TOKENS_FF,  **SAMPLING_FF)

    def run_bucket(name: str, params: SamplingParams) -> None:
        bucket = buckets[name]
        if not bucket:
            return
        idxs, prompts = zip(*bucket)
        print(f"[Phase 4] Generating {name}: {len(prompts)} × N={params.n}")
        t0 = time.perf_counter()
        outputs = llm.generate(list(prompts), sampling_params=params,
                               lora_request=lora_request)
        print(f"[Phase 4]   done in {time.perf_counter() - t0:.1f}s")
        for idx, out in zip(idxs, outputs):
            samples_per_idx[idx] = [c.text.strip() for c in out.outputs]

    run_bucket("mcq",       sampling_mcq)
    run_bucket("ff_single", sampling_ff)
    run_bucket("ff_multi",  sampling_ff)

    # Persist samples so we can diagnose failures with diagnose_failures.py.
    os.makedirs("cache", exist_ok=True)
    with open(HOLDOUT_SAMPLES, "w") as f:
        for item, samples in zip(holdout, samples_per_idx):
            if samples:
                f.write(json.dumps({"id": item["id"], "samples": samples}) + "\n")
    print(f"[Phase 4] Holdout samples saved → {HOLDOUT_SAMPLES}")

    counts = {"mcq": [0, 0], "ff_single": [0, 0], "ff_multi": [0, 0]}
    for item, samples in zip(holdout, samples_per_idx):
        qtype = question_type(item)
        counts[qtype][1] += 1
        if not samples:
            continue
        gold = item["answer"]
        picked = pick_representative_response(samples, qtype, judger=judger)

        if qtype == "mcq":
            picked_letter = extract_letter_mcq(picked)
            correct = picked_letter == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            opts = item.get("options") or []
            options_arg = [opts if opts else []] * len(gold_list)
            try:
                correct = judger.auto_judge(
                    pred=picked, gold=gold_list, options=options_arg)
            except Exception:
                correct = False

        if correct:
            counts[qtype][0] += 1

    print("\n[Phase 4] ────── HOLDOUT EVAL ──────")
    total_c, total_n = 0, 0
    for qtype in ("mcq", "ff_single", "ff_multi"):
        c, n = counts[qtype]
        total_c += c
        total_n += n
        pct = (100 * c / n) if n else 0
        print(f"[Phase 4]   {qtype:10s}: {c:3d}/{n:3d}  ({pct:5.1f}%)")
    pct = (100 * total_c / total_n) if total_n else 0
    print(f"[Phase 4]   {'overall':10s}: {total_c:3d}/{total_n:3d}  ({pct:5.1f}%)")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="Path to public.jsonl for baseline rollouts + rejection sampling")
    p.add_argument("--limit", type=int, default=None,
                   help="Only do baseline rollouts on first N questions (default: full set)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Reuse cache/baseline_samples.jsonl instead of generating")
    p.add_argument("--skip-train", action="store_true",
                   help="Stop after building train_dataset (no QLoRA pass)")
    p.add_argument("--eval-only", action="store_true",
                   help="Run only phase 4 (eval) — assumes ./qlora_adapter exists")
    p.add_argument("--no-lora", action="store_true",
                   help="With --eval-only: smoke test on holdout with BASE model "
                        "only. Useful after editing system prompts — no LoRA "
                        "adapter needed.")
    p.add_argument("--eval-baseline", action="store_true",
                   help="Score cache/baseline_samples.jsonl against gold (CPU only)")
    p.add_argument("--eval-baseline-holdout", action="store_true",
                   help="Like --eval-baseline but only on the 40 holdout ids "
                        "(apples-to-apples comparison with --eval-only output)")
    args = p.parse_args()

    t_start = time.perf_counter()

    if args.eval_baseline or args.eval_baseline_holdout:
        phase_eval_baseline(args.data_path, holdout_only=args.eval_baseline_holdout)
        print(f"[Main] Baseline eval done in {(time.perf_counter() - t_start):.1f} s")
        return

    if args.eval_only:
        phase_eval(args.data_path, use_lora=not args.no_lora)
        print(f"[Main] Eval done in {(time.perf_counter() - t_start)/60:.1f} min")
        return

    if args.skip_baseline:
        print("[Main] Skipping Phase 1 (baseline rollouts)")
    else:
        phase_baseline(args.data_path, args.limit)
        print(f"[Main] Phase 1 done in {(time.perf_counter() - t_start)/60:.1f} min")

    t1 = time.perf_counter()
    train_dataset = phase_build_dataset(args.data_path)
    print(f"[Main] Phase 2 done in {(time.perf_counter() - t1):.1f} s")

    if args.skip_train:
        print("[Main] --skip-train set; stopping before QLoRA pass.")
        return

    t2 = time.perf_counter()
    phase_train(train_dataset)
    print(f"[Main] Phase 3 done in {(time.perf_counter() - t2)/60:.1f} min")

    print(f"[Main] Phases 1-3 done in {(time.perf_counter() - t_start)/60:.1f} min")
    print(f"[Main] Run `python train.py --eval-only` next for holdout evaluation.")


if __name__ == "__main__":
    main()
