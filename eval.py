"""Main eval loop and score reporting for OdiaBench.

Typical usage
-------------
1. Define a ``generate_fn`` that wraps your quantised model:

    def my_generate(prompts: list[str]) -> list[str]:
        # tokenise, run model, decode
        ...
        return decoded_strings

2. Run one benchmark:

    from odia_eval import run_eval, score_report
    results = run_eval("gsm8k", my_generate, n_samples=200, seed=42)
    print(score_report(results))

3. Run all five benchmarks:

    from odia_eval import run_all, full_report
    all_results = run_all(my_generate, n_samples=200)
    print(full_report(all_results))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .analysis import accuracy_with_ci, chance_level
from .datasets import BENCHMARKS, DEFAULT_HF_REPOS, load_benchmark
from .prompts import build_prompt
from .scorers import ScoreResult, score

GenerateFn = Callable[[list[str]], list[str]]


def _strip_prompt(prediction: str, prompt: str) -> str:
    """Return only the model-generated tail of ``prediction``.

    Many ``generate_fn`` implementations return ``prompt + completion``
    (e.g. ``tokenizer.batch_decode(model.generate(...))`` without slicing
    off the input ids).  The benchmark prompts contain ``"A: ..."``,
    ``"1: ..."`` etc., which would otherwise win against the regex
    extractors in :mod:`odia_eval.scorers` — so we strip the prompt here.

    Strategy:
    1. Exact prefix match (fast path).
    2. Substring search for a long-enough tail of the prompt (handles
       tokenizers that inject BOS text or normalise whitespace).
    3. If neither hits, assume the generator already stripped the prompt
       and return ``prediction`` unchanged.
    """
    if prediction.startswith(prompt):
        return prediction[len(prompt):]
    tail = prompt[-60:] if len(prompt) >= 60 else prompt
    idx = prediction.rfind(tail)
    if idx >= 0:
        return prediction[idx + len(tail):]
    return prediction


# ---------------------------------------------------------------------------
# Row-level result
# ---------------------------------------------------------------------------

class EvalRow:
    """One evaluated example."""

    __slots__ = ("benchmark", "row_id", "prompt", "score_result")

    def __init__(
        self,
        benchmark: str,
        row_id: int,
        prompt: str,
        score_result: ScoreResult,
    ) -> None:
        self.benchmark = benchmark
        self.row_id = row_id
        self.prompt = prompt
        self.score_result = score_result

    @property
    def correct(self) -> bool:
        return self.score_result.correct

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark":  self.benchmark,
            "id":         self.row_id,
            "correct":    self.correct,
            "extracted":  self.score_result.extracted,
            "gold":       self.score_result.gold,
            "prediction": self.score_result.prediction,
        }


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

def run_eval(
    benchmark: str,
    generate_fn: GenerateFn,
    *,
    n_samples: int | None = None,
    seed: int = 42,
    batch_size: int = 64,
    data_dir: str | Path | None = None,
    hf_repo: str | None = None,
    verbose: bool = True,
) -> list[EvalRow]:
    """Evaluate ``generate_fn`` on one OdiaBench benchmark.

    Parameters
    ----------
    benchmark:
        One of ``"gsm8k"``, ``"arc"``, ``"truthfulqa"``, ``"winogrande"``,
        ``"hellaswag"``.
    generate_fn:
        ``f(prompts: list[str]) -> list[str]``.  Receives a batch of prompt
        strings, returns a same-length list of decoded model outputs.
        Use ``batch_size`` to control how many prompts are batched per call.
    n_samples:
        Number of rows to evaluate.  ``None`` = full split.
    seed:
        Random seed for sampling (ignored when ``n_samples`` is ``None``).
    batch_size:
        How many prompts to pass to ``generate_fn`` at once — applied
        uniformly across every benchmark.  Default ``64`` works on a
        16 GB T4 with a 2B 4-bit model; bump to ``128`` for 24 GB
        cards (L4/A10G) or ``256`` for ≥40 GB (A100/H100).  Set to ``1``
        when wrapping a non-batching API.
    data_dir:
        Override the default local data directory.
    hf_repo:
        HuggingFace repo id to pull data from instead of local files.
        Pass ``"default"`` to resolve against the canonical
        ``tripathysagar/odia-*`` repos.
    verbose:
        Print a progress line after every batch.

    Returns
    -------
    list[EvalRow]
        One entry per evaluated row.
    """
    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")

    rows = load_benchmark(
        benchmark,
        n_samples=n_samples,
        seed=seed,
        data_dir=data_dir,
        hf_repo=hf_repo,
    )
    total = len(rows)
    results: list[EvalRow] = []

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = [build_prompt(benchmark, row) for row in batch_rows]
        predictions = generate_fn(prompts)

        for row, prompt, pred in zip(batch_rows, prompts, predictions):
            completion = _strip_prompt(pred, prompt)
            sr = score(benchmark, completion, row)
            results.append(
                EvalRow(
                    benchmark=benchmark,
                    row_id=row["id"],
                    prompt=prompt,
                    score_result=sr,
                )
            )

        if verbose:
            done = min(batch_start + batch_size, total)
            correct_so_far = sum(r.correct for r in results)
            pct = 100 * correct_so_far / len(results)
            print(
                f"[{benchmark}] {done}/{total}  "
                f"acc so far: {correct_so_far}/{len(results)} ({pct:.1f}%)"
            )

    return results


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------

def run_all(
    generate_fn: GenerateFn,
    *,
    n_samples: int | None = None,
    seed: int = 42,
    batch_size: int = 64,
    data_dir: str | Path | None = None,
    hf_repos: dict[str, str] | str | None = None,
    verbose: bool = True,
    skip: list[str] | None = None,
) -> dict[str, list[EvalRow]]:
    """Run all five benchmarks sequentially.

    Parameters
    ----------
    skip:
        Benchmark names to skip (e.g. ``["hellaswag"]`` if you want to
        skip the most expensive one for a quick smoke-test).
    hf_repos:
        * ``None`` — load every benchmark from local files (default).
        * ``"default"`` — load every benchmark from the canonical
          ``tripathysagar/odia-*`` repos on HF Hub.
        * ``dict`` — per-benchmark mapping; benchmarks absent from the
          dict fall back to local files.

    Returns
    -------
    dict mapping benchmark name → list[EvalRow]
    """
    if hf_repos == "default":
        hf_repos = dict(DEFAULT_HF_REPOS)
    hf_repos = hf_repos or {}
    skip = skip or []
    all_results: dict[str, list[EvalRow]] = {}

    for name in BENCHMARKS:
        if name in skip:
            if verbose:
                print(f"[{name}] skipped")
            continue
        all_results[name] = run_eval(
            name,
            generate_fn,
            n_samples=n_samples,
            seed=seed,
            batch_size=batch_size,
            data_dir=data_dir,
            hf_repo=hf_repos.get(name),
            verbose=verbose,
        )

    return all_results


# ---------------------------------------------------------------------------
# score_report / full_report
# ---------------------------------------------------------------------------

def _format_row(name: str, results: list[EvalRow], alpha: float = 0.05) -> str:
    acc, lo, hi, correct, total = accuracy_with_ci(results, alpha=alpha)
    chance = chance_level(name)
    delta = (acc - chance) * 100
    delta_str = f"{delta:+5.1f}" if not (chance != chance) else "  n/a"  # nan-safe
    return (
        f"{name:<12s} {correct:>4}/{total:<4}  "
        f"{acc*100:5.1f}%  [{lo*100:5.1f}–{hi*100:5.1f}]  "
        f"chance {chance*100:4.1f}%  Δ {delta_str}"
    )


def score_report(results: list[EvalRow], alpha: float = 0.05) -> str:
    """Return a one-line accuracy summary with Wilson CI and chance delta.

    Example output::

        gsm8k          42/200   21.0%  [15.9–27.1]  chance  0.0%  Δ +21.0
    """
    if not results:
        return "(no results)"
    return _format_row(results[0].benchmark, results, alpha=alpha)


def full_report(
    all_results: dict[str, list[EvalRow]],
    alpha: float = 0.05,
) -> str:
    """Return a multi-line accuracy table with Wilson CIs + chance deltas.

    The ``MEAN`` row averages the per-benchmark accuracies (unweighted)
    and reports the gap above the average chance level so a model that
    beats random on every task shows a positive Δ even if its absolute
    accuracy is modest.

    Example output::

        OdiaBench Evaluation Results  (95% Wilson CI)
        ============================================================
        arc           61/200   30.5%  [24.6–37.1]  chance 25.1%  Δ  +5.4
        gsm8k         42/200   21.0%  [15.9–27.1]  chance  0.0%  Δ +21.0
        hellaswag     55/200   27.5%  [21.7–34.1]  chance 25.0%  Δ  +2.5
        truthfulqa    88/200   44.0%  [37.3–50.9]  chance 22.6%  Δ +21.4
        winogrande    99/200   49.5%  [42.6–56.4]  chance 50.0%  Δ  -0.5
        ------------------------------------------------------------
        MEAN                   34.5%                                Δ +9.9
    """
    title = f"OdiaBench Evaluation Results  ({int((1-alpha)*100)}% Wilson CI)"
    lines = [title, "=" * 60]
    accs: list[float] = []
    chances: list[float] = []
    for name in sorted(all_results):
        rows = all_results[name]
        if not rows:
            continue
        acc, _, _, _, _ = accuracy_with_ci(rows, alpha=alpha)
        accs.append(acc * 100)
        chances.append(chance_level(name) * 100)
        lines.append(_format_row(name, rows, alpha=alpha))
    if accs:
        mean_acc = sum(accs) / len(accs)
        mean_chance = sum(chances) / len(chances)
        mean_delta = mean_acc - mean_chance
        lines.append("-" * 60)
        lines.append(
            f"{'MEAN':<12s} {'':>9s}  {mean_acc:5.1f}%  "
            f"{'':<14s}  {'':<13s}  Δ {mean_delta:+5.1f}"
        )
    return "\n".join(lines)


def to_records(all_results: dict[str, list[EvalRow]]) -> list[dict[str, Any]]:
    """Flatten all results to a list of dicts (convenient for pandas / jsonl)."""
    records: list[dict[str, Any]] = []
    for rows in all_results.values():
        records.extend(r.to_dict() for r in rows)
    return records
