"""Prompt builders for OdiaBench evaluation.

Each ``build_prompt_<name>(row, *, reasoning=False)`` function takes one
dataset row (a plain dict) and returns a single string ready to be fed
to a model.

Design principles
-----------------
* Prompts use Odia-language instructions so the model is tested in the
  target language end-to-end.
* For MCQ tasks the answer choices are presented inside the prompt so
  greedy / beam decoding can produce a letter (A/B/C/D) or digit (1/2)
  rather than a free-form answer.
* For TruthfulQA and Winogrande, the choices are in English (they were
  not translated in the pipeline); the Odia question is paired with the
  English choices, which is intentional — it tests cross-lingual
  instruction-following.
* HellaSwag endings are also in English (only the gold ending was
  translated); the same cross-lingual framing applies.

Reasoning prompts
-----------------
Passing ``reasoning=True`` wraps every prompt with an Odia instruction
that asks the model to **think step-by-step inside ``<think>...</think>``
and put only the final answer inside ``\\boxed{...}``**.  This gives the
model "scratchpad" tokens to compute before committing to an answer
(test-time reasoning) and gives the scorers a deterministic place to
look for the final answer.  The scorers in :mod:`odia_eval.scorers`
already prefer ``\\boxed{...}`` content when it is present, so reasoning
prompts and plain prompts share the same scoring path.
"""
from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# Reasoning prefix
# ---------------------------------------------------------------------------
#
# Wrapper instruction (Odia) that asks the model to:
#   1. think step-by-step inside <think> ... </think>
#   2. then write *only* the final answer inside \boxed{...}
#
# The <think> tag and \boxed{} convention are deliberately kept in English /
# LaTeX because most modern instruction-tuned LLMs (DeepSeek-R1, Qwen2.5,
# Llama-3.1, etc.) are trained on these markers and emit them reliably even
# when the surrounding instruction is in Odia.
_REASONING_PREFIX_OD = (
    "ତୁମେ ପ୍ରଥମେ <think> ଓ </think> ଭିତରେ ଧାପ ଧାପ ଭାବରେ ଚିନ୍ତା କର "
    "(ଯେତିକି ଆବଶ୍ୟକ ସେତିକି ଲେଖ), ଶେଷରେ କେବଳ ଚୂଡାନ୍ତ ଉତ୍ତରଟି "
    "\\boxed{...} ଭିତରେ ଲେଖ।  \\boxed{} ଭିତରେ କୌଣସି ବ୍ୟାଖ୍ୟା ନ ରଖି "
    "କେବଳ ଉତ୍ତର (ସଂଖ୍ୟା ବା ବିକଳ୍ପ ଅକ୍ଷର) ରଖ।\n\n"
)


def _wrap_reasoning(prompt: str) -> str:
    """Prepend the Odia reasoning instruction to ``prompt``."""
    return _REASONING_PREFIX_OD + prompt


def truthfulqa_permutation(row_id: int, n: int) -> list[int]:
    """Deterministic shuffle of TruthfulQA choice indices.

    The HuggingFace TruthfulQA MC1 split stores the correct choice at
    position 0 in every row, which makes a naive prompt builder yield
    ``gold == "A"`` for every question (trivially gamed).  This helper
    returns a permutation seeded by ``row_id`` so both the prompt builder
    and the scorer see the *same* shuffled order without having to pass
    state between them.
    """
    rng = random.Random(f"odia_eval.truthfulqa:{row_id}")
    perm = list(range(n))
    rng.shuffle(perm)
    return perm

# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def build_prompt_gsm8k(row: dict[str, Any], *, reasoning: bool = False) -> str:
    """Chain-of-thought math prompt in Odia.

    The model is asked to reason step-by-step and end with ``#### N``
    (or ``\\boxed{N}`` when ``reasoning=True``).
    """
    prompt = (
        "ନିମ୍ନ ଗଣିତ ପ୍ରଶ୍ନଟି ପଢ଼ ଏବଂ ଧାପ ଧାପ ଭାବରେ ଉତ୍ତର ଦିଅ।\n"
        "ଆପଣଙ୍କ ଉତ୍ତର ଶେଷ ଲାଇନରେ #### N (N = ଉତ୍ତର ସଂଖ୍ୟା) ଆକାରରେ ସାରନ୍ତୁ।\n\n"
        f"ପ୍ରଶ୍ନ: {row['odia_question']}\n"
        "ଉତ୍ତର:"
    )
    return _wrap_reasoning(prompt) if reasoning else prompt


# ---------------------------------------------------------------------------
# ARC
# ---------------------------------------------------------------------------

def build_prompt_arc(row: dict[str, Any], *, reasoning: bool = False) -> str:
    """Multiple-choice science question in Odia.

    ``odia_answer`` already contains the formatted choice block:
    ``A: <text>\\nB: <text>\\nC: <text>\\nD: <text>``
    """
    prompt = (
        "ନିମ୍ନ ବିଜ୍ଞାନ ପ୍ରଶ୍ନ ପାଇଁ ସଠିକ ଉତ୍ତର ବାଛ (A, B, C ବା D)।\n\n"
        f"ପ୍ରଶ୍ନ: {row['odia_question']}\n\n"
        f"{row['odia_answer']}\n\n"
        "ସଠିକ ଉତ୍ତର:"
    )
    return _wrap_reasoning(prompt) if reasoning else prompt


# ---------------------------------------------------------------------------
# TruthfulQA (multiple-choice)
# ---------------------------------------------------------------------------

def build_prompt_truthfulqa(row: dict[str, Any], *, reasoning: bool = False) -> str:
    """TruthfulQA MC prompt.

    The Odia question is paired with English choices (choices were not
    translated in the pipeline).  Choices are **shuffled deterministically
    by row id** because the upstream dataset stores the correct answer at
    index 0 for every row — without shuffling, ``gold`` would always be
    "A" and the benchmark would be trivially gamed.  The scorer replays
    the same permutation so no state needs to travel between them.
    Labels are A, B, C, ... up to the choice count (max 13 in TruthfulQA).
    """
    choices: list[str] = row["mc1_choices"]
    perm = truthfulqa_permutation(row["id"], len(choices))
    shuffled = [choices[i] for i in perm]
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_block = "\n".join(
        f"{labels[i]}: {choice}" for i, choice in enumerate(shuffled)
    )
    prompt = (
        "ନିମ୍ନ ପ୍ରଶ୍ନ ପାଇଁ ସଠିକ ଉତ୍ତର ବାଛ।\n\n"
        f"ପ୍ରଶ୍ନ: {row['odia_question']}\n\n"
        f"{choice_block}\n\n"
        "ସଠିକ ଉତ୍ତର (ଅକ୍ଷର):"
    )
    return _wrap_reasoning(prompt) if reasoning else prompt


# ---------------------------------------------------------------------------
# Winogrande
# ---------------------------------------------------------------------------

def build_prompt_winogrande(row: dict[str, Any], *, reasoning: bool = False) -> str:
    """Coreference / fill-in-the-blank prompt.

    The Odia sentence has ``_`` as the blank; English options are presented
    as numbered choices.
    """
    prompt = (
        "ଶୂନ୍ୟସ୍ଥାନ (_) ପୂରଣ ପାଇଁ ସଠିକ ବିକଳ୍ପ ବାଛ (1 ବା 2)।\n\n"
        f"ବାକ୍ୟ: {row['odia_question']}\n\n"
        f"1: {row['option1']}\n"
        f"2: {row['option2']}\n\n"
        "ସଠିକ ଉତ୍ତର (1 ବା 2):"
    )
    return _wrap_reasoning(prompt) if reasoning else prompt


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

def build_prompt_hellaswag(row: dict[str, Any], *, reasoning: bool = False) -> str:
    """Sentence completion prompt.

    The Odia context is paired with the four English endings (endings were
    not fully translated in the pipeline — only the gold ending was).
    Endings are labelled A-D.
    """
    endings: list[str] = row["all_endings"]
    labels = "ABCD"
    ending_block = "\n".join(
        f"{labels[i]}: {ending}" for i, ending in enumerate(endings)
    )
    prompt = (
        "ନିମ୍ନ ଅନୁଚ୍ଛେଦ ପାଇଁ ସବୁଠୁ ଉପଯୁକ୍ତ ସମାପ୍ତି ବାଛ (A, B, C ବା D)।\n\n"
        f"ପ୍ରସଙ୍ଗ: {row['odia_question']}\n\n"
        f"{ending_block}\n\n"
        "ସଠିକ ଉତ୍ତର:"
    )
    return _wrap_reasoning(prompt) if reasoning else prompt


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_BUILDERS = {
    "gsm8k":      build_prompt_gsm8k,
    "arc":        build_prompt_arc,
    "truthfulqa": build_prompt_truthfulqa,
    "winogrande": build_prompt_winogrande,
    "hellaswag":  build_prompt_hellaswag,
}


def build_prompt(
    benchmark: str,
    row: dict[str, Any],
    *,
    reasoning: bool = False,
) -> str:
    """Build a prompt for ``row`` from the named benchmark.

    Parameters
    ----------
    benchmark:
        One of :data:`odia_eval.BENCHMARKS`.
    row:
        A dataset row dict produced by :func:`odia_eval.load_benchmark`.
    reasoning:
        If ``True``, prepend an Odia instruction asking the model to
        think inside ``<think>...</think>`` and emit the final answer in
        ``\\boxed{...}``.  Pairs with the boxed-aware scorers in
        :mod:`odia_eval.scorers` so the answer is extracted from
        ``\\boxed{...}`` whenever the model honours the format.
    """
    if benchmark not in _BUILDERS:
        raise KeyError(f"unknown benchmark {benchmark!r}; expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[benchmark](row, reasoning=reasoning)
