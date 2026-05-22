"""Scoring functions for OdiaBench evaluation.

Every ``score_<name>(prediction, row)`` function takes:
  - ``prediction``: the raw string produced by the model
  - ``row``: the original dataset dict

Returns a ``ScoreResult`` with ``correct`` (bool) and ``extracted`` (the
value actually parsed from the prediction for debugging).

All extractors are lenient: they search the full prediction string rather
than requiring the answer to appear at a specific position, so both
chain-of-thought responses and short answers are handled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .prompts import truthfulqa_permutation

_FOUR_RE = re.compile(r"####\s*(-?\d+(?:[.,]\d+)?)")
_LETTER_AD_RE = re.compile(r"\b([A-Da-d])\b")        # ARC / HellaSwag
_LETTER_AE_RE = re.compile(r"\b([A-Ea-e])\b")        # ARC (one row has answerKey=E)
_DIGIT12_RE = re.compile(r"\b([12])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
# ``\boxed{...}`` (LaTeX) — used by reasoning prompts and many CoT-tuned
# models.  We deliberately do **not** support nested braces (benchmark
# answers are single tokens) so ``[^{}]*?`` is enough.
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]*?)\s*\}", re.DOTALL)

# Odia numerals → ASCII
_ODIA_DIGIT_TABLE = str.maketrans("୦୧୨୩୪୫୬୭୮୯", "0123456789")

# Odia digit words → ASCII (covers the most common spellings models emit
# for Winogrande, which only ever needs 1 or 2)
_ODIA_DIGIT_WORDS = {
    "ଗୋଟିଏ": "1", "ଏକ": "1", "ପ୍ରଥମ": "1",
    "ଦୁଇ": "2", "ଦୁଇଟି": "2", "ଦ୍ୱିତୀୟ": "2",
}


@dataclass
class ScoreResult:
    correct: bool
    extracted: str          # what was parsed from the prediction
    gold: str               # the expected value
    prediction: str         # raw model output (truncated to 512 chars for storage)

    def __repr__(self) -> str:
        mark = "✓" if self.correct else "✗"
        return f"ScoreResult({mark} extracted={self.extracted!r} gold={self.gold!r})"


def _normalise(text: str) -> str:
    """Strip Odia numerals and normalise comma-separated numbers."""
    text = text.translate(_ODIA_DIGIT_TABLE)
    text = text.replace(",", "")   # 1,000 → 1000
    return text.strip()


def _normalise_digit_words(text: str) -> str:
    """Replace Odia number-words (ଗୋଟିଏ, ଦୁଇ, ...) with ASCII digits.

    Used only by the Winogrande scorer, which has a 1/2 answer space.
    """
    for word, digit in _ODIA_DIGIT_WORDS.items():
        text = text.replace(word, f" {digit} ")
    return text


def _letter_re(n: int) -> re.Pattern[str]:
    """Build a ``\\b[A-X]\\b`` regex sized to ``n`` choices (max 26)."""
    n = max(1, min(n, 26))
    last_upper = chr(ord("A") + n - 1)
    last_lower = chr(ord("a") + n - 1)
    return re.compile(rf"\b([A-{last_upper}a-{last_lower}])\b")


def _extract_boxed(text: str) -> str:
    """Return the content of the **last** ``\\boxed{...}`` in ``text``.

    Reasoning prompts (and many CoT-tuned models) wrap the final answer
    in ``\\boxed{...}``.  We pick the *last* **non-empty** occurrence so
    any "scratch" boxes in the model's intermediate work are ignored,
    and a trailing empty ``\\boxed{}`` placeholder in the prompt text
    (from the reasoning instruction) does not suppress fallback to legacy
    extractors.  Returns ``""`` if no non-empty box exists.
    """
    matches = [m.strip() for m in _BOXED_RE.findall(text)]
    non_empty = [m for m in matches if m]
    return non_empty[-1] if non_empty else ""


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def score_gsm8k(prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Extract the number after ``####`` and compare to the gold answer.

    Extraction precedence:
    1. The last ``\\boxed{...}`` in the prediction (reasoning prompts).
    2. The number after ``####`` (legacy CoT format).
    3. The last standalone number anywhere in the prediction.
    """
    pred_norm = _normalise(prediction)
    gold_norm = _normalise(row["odia_answer"])

    gold_m = _FOUR_RE.search(gold_norm)
    gold_val = gold_m.group(1).replace(",", "") if gold_m else ""

    extracted = ""
    boxed = _extract_boxed(pred_norm)
    if boxed:
        nums_in_box = _NUMBER_RE.findall(boxed)
        if nums_in_box:
            extracted = nums_in_box[-1].replace(",", "")
    if not extracted:
        pred_m = _FOUR_RE.search(pred_norm)
        if pred_m:
            extracted = pred_m.group(1).replace(",", "")
    if not extracted:
        nums = _NUMBER_RE.findall(pred_norm)
        extracted = nums[-1].replace(",", "") if nums else ""

    # Compare numerically so "1.50" == "1.5" and "18.0" == "18"
    def _num_eq(a: str, b: str) -> bool:
        try:
            return float(a) == float(b)
        except ValueError:
            return a == b

    correct = bool(extracted) and _num_eq(extracted, gold_val)
    return ScoreResult(
        correct=correct,
        extracted=extracted,
        gold=gold_val,
        prediction=prediction[:512],
    )


# ---------------------------------------------------------------------------
# ARC
# ---------------------------------------------------------------------------

_ARC_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def score_arc(prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Extract the first A-E letter from the prediction.

    ARC datasets use both letter keys (A-E) and numeric keys (1-5); both
    are normalised to letters before comparison.  The regex is widened
    to E so the (rare) 5-choice rows are still scoreable.

    If the prediction contains ``\\boxed{...}``, the letter is taken
    from inside the **last** box (reasoning-prompt path); otherwise the
    full prediction is searched.
    """
    raw_key = str(row["answerKey"]).strip().upper()
    gold = _ARC_NUM_TO_LETTER.get(raw_key, raw_key)

    pred_norm = _normalise(prediction)
    boxed = _extract_boxed(pred_norm)
    search_text = boxed if boxed else pred_norm
    m = _LETTER_AE_RE.search(search_text)
    extracted = m.group(1).upper() if m else ""

    return ScoreResult(
        correct=extracted == gold,
        extracted=extracted,
        gold=gold,
        prediction=prediction[:512],
    )


# ---------------------------------------------------------------------------
# TruthfulQA (multiple-choice)
# ---------------------------------------------------------------------------

def score_truthfulqa(prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Map predicted letter back to choice index; compare to gold label.

    The HuggingFace TruthfulQA dataset stores the correct choice at index 0
    for every row, so the prompt builder shuffles ``mc1_choices``
    deterministically (seeded by ``row['id']``).  We replay the same
    permutation here to recover the gold letter under the shuffled order.

    The letter regex is sized to the actual number of choices (up to 13
    for TruthfulQA), so answers labelled E-M are extractable too.
    """
    labels: list[int] = row["mc1_labels"]
    try:
        original_gold_idx = labels.index(1)
    except ValueError:
        original_gold_idx = 0

    n = len(labels)
    perm = truthfulqa_permutation(row["id"], n)
    shuffled_gold_idx = perm.index(original_gold_idx)
    gold_letter = chr(ord("A") + shuffled_gold_idx)

    pred_norm = _normalise(prediction)
    boxed = _extract_boxed(pred_norm)
    search_text = boxed if boxed else pred_norm
    m = _letter_re(n).search(search_text)
    extracted = m.group(1).upper() if m else ""

    return ScoreResult(
        correct=extracted == gold_letter,
        extracted=extracted,
        gold=gold_letter,
        prediction=prediction[:512],
    )


# ---------------------------------------------------------------------------
# Winogrande
# ---------------------------------------------------------------------------

def score_winogrande(prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Extract 1 or 2 from prediction; compare to ``answer_idx``.

    The prediction is run through :func:`_normalise` (Odia → ASCII digits)
    and :func:`_normalise_digit_words` (ଗୋଟିଏ/ଦୁଇ → 1/2) before extraction
    so Odia-language outputs are scored fairly.

    Rows with an empty ``answer_idx`` (e.g. upstream Winogrande test, which
    hides gold labels) are unscorable and always return ``correct=False``
    rather than fake-matching an empty extraction against an empty gold.
    """
    gold = str(row.get("answer_idx", "")).strip()
    if not gold:
        return ScoreResult(
            correct=False,
            extracted="",
            gold="",
            prediction=prediction[:512],
        )

    pred_norm = _normalise_digit_words(_normalise(prediction))
    boxed = _extract_boxed(pred_norm)
    search_text = boxed if boxed else pred_norm
    m = _DIGIT12_RE.search(search_text)
    extracted = m.group(1) if m else ""

    return ScoreResult(
        correct=extracted == gold,
        extracted=extracted,
        gold=gold,
        prediction=prediction[:512],
    )


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def score_hellaswag(prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Extract A/B/C/D from prediction; compare to ``label`` (0-indexed).

    Rows with an empty / missing ``label`` (e.g. upstream HellaSwag test,
    which hides gold labels) are unscorable and always return
    ``correct=False`` rather than silently defaulting gold to ``"A"`` and
    rewarding any model that emits ``"A"``.
    """
    raw_label = row.get("label")
    if raw_label in ("", None):
        return ScoreResult(
            correct=False,
            extracted="",
            gold="",
            prediction=prediction[:512],
        )
    try:
        gold_idx = int(raw_label)
    except (ValueError, TypeError):
        return ScoreResult(
            correct=False,
            extracted="",
            gold="",
            prediction=prediction[:512],
        )
    gold_letter = "ABCD"[gold_idx]

    pred_norm = _normalise(prediction)
    boxed = _extract_boxed(pred_norm)
    search_text = boxed if boxed else pred_norm
    m = _LETTER_AD_RE.search(search_text)
    extracted = m.group(1).upper() if m else ""

    return ScoreResult(
        correct=extracted == gold_letter,
        extracted=extracted,
        gold=gold_letter,
        prediction=prediction[:512],
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SCORERS = {
    "gsm8k":      score_gsm8k,
    "arc":        score_arc,
    "truthfulqa": score_truthfulqa,
    "winogrande": score_winogrande,
    "hellaswag":  score_hellaswag,
}


def score(benchmark: str, prediction: str, row: dict[str, Any]) -> ScoreResult:
    """Score one prediction for the named benchmark."""
    if benchmark not in _SCORERS:
        raise KeyError(f"unknown benchmark {benchmark!r}; expected one of {sorted(_SCORERS)}")
    return _SCORERS[benchmark](prediction, row)
