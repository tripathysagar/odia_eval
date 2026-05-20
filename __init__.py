"""odia_eval — OdiaBench evaluation utilities.

Install
-------
``odia_eval`` itself has **no required dependencies** — it is pure
stdlib.  Optional extras live in ``odia_eval/requirements.txt``:

* ``datasets``                — HuggingFace Hub loading via ``hf_repo=``
* ``pandas``                  — ``to_records`` → ``DataFrame`` export
* ``torch`` / ``transformers``  / ``accelerate`` / ``bitsandbytes``
                              — example 4-bit reference model only

To install every optional extra at once::

    pip install -r odia_eval/requirements.txt

Quick start
-----------
    import sys
    sys.path.insert(0, "/kaggle/working/OdiaBench")   # after git clone

    from odia_eval import run_eval, run_all, score_report, full_report

    # Define your model's generate function.
    # NOTE: slice off the prompt tokens before decoding so the prediction
    # contains only the completion — otherwise the scorers' regex
    # extractors will match letters/digits from the echoed prompt.
    # (run_eval also strips the prompt as a safety net, but doing it
    # here is cleaner and avoids ambiguity for tokenizers that mutate
    # whitespace or inject a BOS token text.)
    def my_generate(prompts: list[str]) -> list[str]:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256)
        gen_ids = [o[len(i):] for o, i in zip(out, enc["input_ids"])]
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    # Evaluate one benchmark (200 random samples)
    results = run_eval("gsm8k", my_generate, n_samples=200)
    print(score_report(results))

    # Evaluate all five benchmarks
    all_results = run_all(my_generate, n_samples=200, skip=["hellaswag"])
    print(full_report(all_results))
"""
from .datasets import BENCHMARKS, DEFAULT_HF_REPOS, load_benchmark, load_all
from .prompts import build_prompt, truthfulqa_permutation
from .scorers import score, ScoreResult
from .eval import (
    EvalRow,
    GenerateFn,
    run_eval,
    run_all,
    score_report,
    full_report,
    to_records,
    _strip_prompt,
)
from .analysis import (
    CHANCE_LEVELS,
    chance_level,
    wilson_ci,
    accuracy_with_ci,
    show_failures,
)
from .io import save_results, load_results

# 0.2.0 — TruthfulQA choices are now deterministically shuffled by row id
# and several scorers were widened (ARC A-E, TFQA A-M, Odia digit/word
# normalisation for Winogrande).  Any results stored under 0.1.0 should
# be regenerated before being compared to 0.2.0+ runs.
__version__ = "0.2.0"

__all__ = [
    # datasets
    "BENCHMARKS",
    "DEFAULT_HF_REPOS",
    "load_benchmark",
    "load_all",
    # prompts
    "build_prompt",
    "truthfulqa_permutation",
    # scorers
    "score",
    "ScoreResult",
    # eval
    "EvalRow",
    "GenerateFn",
    "run_eval",
    "run_all",
    "score_report",
    "full_report",
    "to_records",
    "_strip_prompt",
    # analysis
    "CHANCE_LEVELS",
    "chance_level",
    "wilson_ci",
    "accuracy_with_ci",
    "show_failures",
    # io
    "save_results",
    "load_results",
]
