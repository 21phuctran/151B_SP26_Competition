"""Inference helpers shared between the training notebook and run_inference.py.

This is the single source of truth for prompt construction, answer extraction,
and self-consistency voting. The notebook should import from this file (rather
than defining everything inline) so the training-time and inference-time
behaviour can't drift apart.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional


# ─── System prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = """
You are solving a math problem for an automatic grader. The grader extracts your answer from \\boxed{} in the post-think text (after </think>) and checks it against gold using sympy with ~1e-6 numeric tolerance.

# CRITICAL — read this first
A truncated trace with no \\boxed{} after </think> SCORES ZERO. Watch your token budget. If you've been thinking for more than ~3000 words OR are unsure of the answer, STOP IMMEDIATELY, write </think>, and commit your best guess in \\boxed{}. A boxed guess scores; a great trace without a boxed answer does not.

# How to think
Reason step by step inside <think>...</think>. Verify each step (arithmetic, signs, domains, units, what variable was actually asked). You may write multiple \\boxed{} expressions inside <think>...</think> while exploring — only the post-think text matters.

# Count the blanks BEFORE you box
Count the number of `[ANS]` placeholders in the question. Your boxed answer must contain EXACTLY that many values, in the same ORDER they appear. If the question has 4 blanks, you write 4 values. If it has 1 blank, 1 value. Wrong count = wrong answer.

# Final answer block (this is everything that gets scored)
After </think>, write a brief 1-3 line summary, then this closing line:

Therefore, the final answer is \\boxed{...}.

If multiple blanks, prefer ONE box with comma-separated values, but multiple \\boxed{} expressions in a row are also acceptable to the grader.

# What goes inside \\boxed{} — by problem cues
The right format depends on what the problem asks for. Read the problem text carefully:

**Problem asks for "exact form", "exact value", "no decimals", or uses symbolic constants (π, e, √, ln):**
→ Use SYMBOLIC form. \\frac{\\sqrt{3}}{2} not 0.866, \\frac{\\pi}{4} not 0.785, \\ln(2) not 0.693, \\arctan(4.76) not 1.36.

**Problem asks for a decimal, "approximate", numeric value, or gives numeric inputs without symbolic constants:**
→ Use DECIMAL with ≥10 significant figures. Write "143.22422923" not "143.22". Never round to 2-3 decimal places EVEN IF the problem says "round to 2 decimals" — the grader uses tight tolerance and aggressive rounding fails.

**When in doubt (problem could go either way):**
→ Output the SYMBOLIC form. It evaluates under sympy and matches both symbolic AND decimal gold answers.

**Multi-part with mixed types:**
→ Each blank uses the form most natural for that part. Comma-separate in order.

# DECIMAL PRECISION — the most common reason answers get rejected

The grader uses ~1e-6 tolerance. Rounding loses points. Concrete examples:

✗ WRONG: \\boxed{2.10}                    ← 3 sig figs, rejected
✓ RIGHT: \\boxed{2.09959521978367}        ← 12 sig figs, accepted

✗ WRONG: \\boxed{12.08}                   ← problem said "round to 2 decimals", model obeyed → rejected
✓ RIGHT: \\boxed{12.0813557729}           ← ignore the "round to N" instruction, keep precision

✗ WRONG: \\boxed{0.34}                    ← 2 sig figs, rejected
✓ RIGHT: \\boxed{0.3402817845}            ← 10 sig figs, accepted

The rule: if your answer is a decimal, COUNT THE DIGITS before boxing. You need at least 10 significant figures. Trailing zeros count: 14.1100000000 is fine; 14.11 is not.

# Format examples

Single value, exact:
Therefore, the final answer is \\boxed{\\frac{\\sqrt{3}}{2}}.

Single value, decimal (≥10 sig figs):
Therefore, the final answer is \\boxed{143.2242292337}.

Multiple values, in order asked:
Therefore, the final answer is \\boxed{3, \\frac{1}{2}, \\sqrt{5}}.

Multiple decimal values (no 2-decimal rounding, even if problem requests):
Therefore, the final answer is \\boxed{62.77777778, 335.92777778, 604.67000000}.

Set (unordered):
Therefore, the final answer is \\boxed{\\{-3, 3\\}}.

Interval:
Therefore, the final answer is \\boxed{(-\\infty, 2) \\cup (2, \\infty)}.

Equation of a curve:
Therefore, the final answer is \\boxed{y = 2x + 1}.

True/False:
Therefore, the final answer is \\boxed{True}.

Yes/No (use the case the problem uses, often lowercase):
Therefore, the final answer is \\boxed{Yes}.

Coordinates:
Therefore, the final answer is \\boxed{(-1, -3)}.

± answers:
Therefore, the final answer is \\boxed{\\pm 3}.

# Pre-box checklist (silently, before closing)
1. **Count check:** number of values in my box == number of [ANS] blanks in the question?
2. **Order check:** are the values in the order the question asks?
3. **What was asked:** did I answer the final quantity, not an intermediate?
4. **Sign check:** correct sign on every value?
5. **Format check:** symbolic for exact-form problems, ≥10 sig figs for numeric problems?
6. **NOT rounded:** if numeric, did I avoid rounding to 2-3 decimals even if asked?

# Hard rules
- The post-think text MUST contain at least one \\boxed{}. Without it, your score is zero.
- Inside the box: values only — no "x =", no \\text{...} wrappers, no units.
- Decimal numbers: ≥10 significant figures, NEVER rounded to 2-3 decimals.
- If you sense token budget running low or you're stuck, immediately write </think> and a \\boxed{best guess}. A guess scores; no answer does not.
""".strip()


SYSTEM_PROMPT_MCQ = """
You are answering a multiple-choice problem for an automatic grader. One of the listed options is correct — your job is to identify the LETTER, not re-derive the answer with full rigor.

# CRITICAL — read this first
A truncated trace with no \\boxed{X} after </think> SCORES ZERO. Watch your token budget. If you've been thinking for more than ~3000 words, OR you've explored multiple approaches without converging, OR you're not sure of the answer — STOP IMMEDIATELY, write </think>, and commit your best guess in \\boxed{X}. Even a random letter scores better than no letter.

# Strategy
- Solve only enough to distinguish the correct option from the distractors. You usually do not need a full derivation.
- When it's faster, plug candidate options back into the problem.
- If your computed result matches one option (numerically or symbolically), pick that option — do not redo the work.
- Eliminate options aggressively: ruling out 3 of 4 = solving.

# Final answer block
After </think>, write a brief 1-2 line justification, then close with exactly:

Therefore, the answer is \\boxed{X}.

Where X is a SINGLE UPPERCASE LETTER. No parentheses, no period inside the box, no option text.

✓ Correct: \\boxed{C}
✗ Wrong:   \\boxed{(C)}, \\boxed{C.}, \\boxed{C) 7}, \\boxed{Option C}, \\boxed{three}

For "select all" problems where multiple letters are correct, output them as a single concatenated string with no separator:
✓ Correct: \\boxed{BCEG}
(Both \\boxed{BCEG} and \\boxed{B, C, E, G} are accepted, but use the concatenated form by default.)

# Pre-box checklist (silently, before closing)
1. **Letter-option match:** does the letter I'm about to box correspond to the option I believe is correct? (Most common error: derive correctly, then box the wrong letter.)
2. **One value per blank:** if the problem has [ANS] markers, count them. Output one letter per blank.

# Hard rules
- The post-think text MUST contain a \\boxed{LETTER}. Without it, your score is zero.
- Box ONLY the letter(s) — no derivation, no values, no commas to numbers.
- If you're stuck or out of budget: \\boxed{your best guess letter} immediately. Random guess > no answer.
""".strip()


# ─── Prompt construction ─────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question.

    Used by both the training notebook (with public.jsonl) and run_inference.py
    (with private.jsonl). Branches on `options` to pick MCQ vs free-form.

    For multi-blank free-form questions, injects an explicit blank-count hint
    into the user prompt — the system prompt's "count the blanks" rule was
    insufficient on long multi-blank problems (15-blank questions were getting
    a single boxed value). Doing the counting in-prompt makes it unmissable.
    """
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    n_blanks = question.count("[ANS]")
    if n_blanks > 1:
        hint = (f"\n\n[REMINDER: this question contains {n_blanks} `[ANS]` blanks. "
                f"Your \\boxed{{}} must contain exactly {n_blanks} comma-separated "
                f"values, in the order the blanks appear.]")
        return SYSTEM_PROMPT_MATH, question + hint
    return SYSTEM_PROMPT_MATH, question


def question_type(item: dict) -> str:
    """Return 'mcq', 'ff_multi', or 'ff_single'.

    For private.jsonl items there's no `answer` field, so ff_multi vs ff_single
    is decided by counting `[ANS]` placeholders in the question.
    """
    if item.get("options"):
        return "mcq"
    ans = item.get("answer")
    if isinstance(ans, list):
        return "ff_multi" if len(ans) > 1 else "ff_single"
    # No answer field (private set) — fall back to placeholder count
    n_ans = item["question"].count("[ANS]")
    return "ff_multi" if n_ans > 1 else "ff_single"


# ─── Answer extraction ───────────────────────────────────────────────────────

_JUNK_ANSWERS = {"", "?", "??", "...", "unsure", "unknown", "none",
                 "n/a", "na", "tbd", "todo"}


def is_junk(ans: Optional[str]) -> bool:
    if ans is None:
        return True
    s = ans.strip().lower()
    return s in _JUNK_ANSWERS or len(s) == 0


def extract_letter_mcq(text: str) -> str:
    """Extract MCQ letter from a sample. Looks at post-think text first."""
    think_end = text.rfind("</think>")
    search = text[think_end + len("</think>"):] if think_end >= 0 else text

    m = re.search(r"\\boxed\{\s*\(?\s*([A-Za-z])\s*[\)\.]?\s*\}", search)
    if m:
        return m.group(1).upper()

    m = re.search(r"answer\s*(?:is|:)\s*\(?\s*([A-Za-z])\s*[\)\.]?", search, re.I)
    if m:
        return m.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", search)
    return matches[-1] if matches else ""


# ─── Voting ──────────────────────────────────────────────────────────────────

def vote_plurality(items):
    """Plurality vote, dropping junk. Returns (winner, vote_count)."""
    items = [x for x in items if not is_junk(x)]
    if not items:
        return "", 0
    counts = Counter(items)
    winner, n = counts.most_common(1)[0]
    return winner, n


def vote_normalized(raw_items, norm_fn: Callable[[str], str]):
    """Vote by normalized form; return representative RAW string.

    Same normalized key wins votes together (\\frac{1}{2} ≡ 0.5 ≡ 1/2), but
    the returned string is the first raw form that mapped to the winner — so
    the output looks like the model's natural output.
    """
    valid = [(r, norm_fn(r)) for r in raw_items if not is_junk(r)]
    if not valid:
        return "", 0, 0
    counts = Counter(k for _, k in valid)
    winning_key, n = counts.most_common(1)[0]
    rep = next(r for r, k in valid if k == winning_key)
    return rep, n, len(counts)


def safe_norm(judger, s: str) -> str:
    """Wrap judger.norm_ans_str so bad parses don't kill the run."""
    try:
        return judger.norm_ans_str(s)
    except Exception:
        return s.strip()


# ─── Choose the response trace to submit ─────────────────────────────────────
#
# The competition CSV needs the FULL response trace (including <think>...</think>),
# not just the voted answer. So after voting, we pick one sample's full text as
# the representative — specifically, the first sample whose extracted answer
# matches the vote winner.

def pick_representative_response(samples: list[str], qtype: str, judger=None) -> str:
    """Return the full text of the sample best representing the self-consistency vote.

    For MCQ: pick first sample whose extracted letter equals the vote winner.
    For FF: pick first sample whose judger-extracted answer normalizes to the
            same form as the vote winner (across all blanks).

    Falls back to the first non-empty sample if voting produces no winner.
    """
    samples = [s for s in samples if s]
    if not samples:
        return ""

    if qtype == "mcq":
        letters = [extract_letter_mcq(s) for s in samples]
        winner, _ = vote_plurality(letters)
        if not winner:
            return samples[0]
        for s, l in zip(samples, letters):
            if l == winner:
                return s
        return samples[0]

    # Free-form: vote on judger-extracted + normalized form.
    if judger is None:
        return samples[0]

    extracted = []
    for s in samples:
        try:
            e = judger.extract_ans(s) or ""
        except Exception:
            e = ""
        extracted.append(e)

    # NOTE: per-blank voting was implemented and tested on 440 ff_multi
    # questions across two datasets; it never flipped a single question in
    # either direction vs whole-answer voting. The theoretical downside
    # (when model has consistent errors at one position, whole-answer
    # plurality can pick the right sample but per-blank locks into the
    # noisy modal) is more common than the theoretical upside. So we
    # ship with whole-answer voting only. See _ff_multi_per_blank_vote
    # (unused) for the implementation if revisiting later.

    normalized = [safe_norm(judger, e) for e in extracted]
    winner, _ = vote_plurality(normalized)
    if not winner:
        return samples[0]
    for s, n in zip(samples, normalized):
        if n == winner:
            return s
    return samples[0]


def _ff_multi_per_blank_vote(samples, extracted, judger):
    """Per-blank voting for FF-multi. Returns the donor sample's text with
    its \\boxed{} rewritten to contain the comma-joined per-position winners,
    or None if per-blank voting can't be applied cleanly.
    """
    # Split each sample's extraction into parts
    parts_list = []
    for e in extracted:
        try:
            parts = judger.split_by_comma(e) if e else []
        except Exception:
            parts = []
        parts_list.append(parts)

    # Determine modal part count
    count_counts = Counter(len(p) for p in parts_list if p)
    if not count_counts:
        return None
    target_count, _ = count_counts.most_common(1)[0]
    if target_count <= 1:
        return None  # not actually multi-blank

    # Keep samples with the modal count
    valid = [(i, parts_list[i]) for i in range(len(samples))
             if len(parts_list[i]) == target_count]
    if len(valid) < 2:
        return None  # not enough samples for voting

    # Vote per position
    winners_norm: list[str] = []
    winners_raw:  list[str] = []
    for pos in range(target_count):
        pos_pairs = [(parts[pos], safe_norm(judger, parts[pos]))
                     for _, parts in valid]
        # Drop junk values from voting (e.g. empty strings)
        valid_pairs = [(r, k) for r, k in pos_pairs if not is_junk(r)]
        if not valid_pairs:
            return None  # can't vote on this position
        counts = Counter(k for _, k in valid_pairs)
        winning_key, _ = counts.most_common(1)[0]
        winners_norm.append(winning_key)
        rep = next(r for r, k in valid_pairs if k == winning_key)
        winners_raw.append(rep)

    # Pick donor: sample whose parts match the most per-position winners
    best_idx, best_match = -1, -1
    for i, parts in valid:
        norm_parts = [safe_norm(judger, p) for p in parts]
        match = sum(1 for np, win in zip(norm_parts, winners_norm) if np == win)
        if match > best_match:
            best_match = match
            best_idx = i

    if best_idx < 0:
        return None

    donor = samples[best_idx]

    # Guard 1: if donor's parts already match the winners exactly, the donor
    # text is already correct — don't rewrite (avoids corrupting samples that
    # use multiple separate \boxed{} expressions in the answer).
    donor_norm = [safe_norm(judger, p) for p in parts_list[best_idx]]
    if donor_norm == winners_norm:
        return donor

    # Otherwise, attempt to rewrite the last \boxed{} with the voted values.
    new_answer = ", ".join(winners_raw)
    rewritten = _rewrite_last_boxed(donor, new_answer)
    if rewritten is donor:
        return donor  # rewrite was unsafe

    # Guard 2: verify the rewrite didn't corrupt extraction (e.g. donor had
    # multiple \boxed{} expressions and we only replaced the last, producing
    # a count mismatch). If extraction now gives a different number of parts
    # than the donor previously did, fall back to the original donor.
    try:
        new_extraction = judger.extract_ans(rewritten) or ""
        new_parts = judger.split_by_comma(new_extraction) if new_extraction else []
    except Exception:
        return donor
    if len(new_parts) != target_count:
        return donor
    return rewritten


def _rewrite_last_boxed(text: str, new_content: str) -> str:
    """Replace the content of the last \\boxed{...} (after </think> if present)
    with new_content. Returns the original text unchanged if the rewrite is
    unsafe (no </think>, no box found, malformed braces)."""
    think_end = text.rfind("</think>")
    if think_end < 0:
        return text
    search_offset = think_end + len("</think>")
    last_box_marker = text.rfind("\\boxed{", search_offset)
    if last_box_marker < 0:
        return text
    brace_start = last_box_marker + len("\\boxed{")
    depth = 1
    i = brace_start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return text  # malformed
    closing_brace = i - 1
    return text[:brace_start] + new_content + text[closing_brace:]
