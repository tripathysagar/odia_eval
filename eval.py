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

from collections import Counter
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
# Test-time compute: majority-vote (self-consistency) aggregator
# ---------------------------------------------------------------------------

def _canonical_extracted(benchmark: str, extracted: str) -> str:
    """Canonicalise an extracted answer for vote-grouping / comparison.

    * ``gsm8k`` — parse as float so ``"1.5"``, ``"1.50"`` and ``"01.5"``
      collapse to the same bucket (``"1.5"``).
    * MCQ benchmarks (``arc``, ``truthfulqa``, ``hellaswag``) — uppercase
      so ``"a"`` and ``"A"`` count as the same vote.
    * ``winogrande`` — digit string, returned as-is after strip.
    * Empty input returns ``""`` (treated as an abstention).
    """
    if not extracted:
        return ""
    s = extracted.strip()
    if benchmark == "gsm8k":
        try:
            v = float(s.replace(",", ""))
            return f"{v:g}"
        except ValueError:
            return s
    if benchmark in ("arc", "truthfulqa", "hellaswag"):
        return s.upper()
    return s


def _aggregate_majority_vote(
    benchmark: str,
    completions: list[str],
    row: dict[str, Any],
) -> ScoreResult:
    """Score ``completions`` independently and return a majority-vote result.

    Self-consistency (Wang et al., 2022) for benchmarks: sample N answers
    per prompt, then pick the answer the model agreed on most often.
    Ties are broken by first-seen order (``Counter.most_common`` is
    stable on equal counts in CPython 3.7+).  Empty extractions are
    abstentions — they never win.  If *every* completion abstains the
    result is marked incorrect with ``extracted=""``.
    """
    if not completions:
        return ScoreResult(correct=False, extracted="", gold="", prediction="")

    per_sample = [score(benchmark, c, row) for c in completions]
    gold = per_sample[0].gold

    keys = [_canonical_extracted(benchmark, sr.extracted) for sr in per_sample]
    non_empty = [k for k in keys if k]

    if non_empty:
        winner_key, _ = Counter(non_empty).most_common(1)[0]
        winner_extracted = next(
            sr.extracted for sr, k in zip(per_sample, keys) if k == winner_key
        )
        correct = (
            bool(winner_key)
            and winner_key == _canonical_extracted(benchmark, gold)
        )
    else:
        winner_extracted = ""
        correct = False

    joined = "\n---\n".join(completions)
    return ScoreResult(
        correct=correct,
        extracted=winner_extracted,
        gold=gold,
        prediction=joined[:512],
    )


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
    reasoning: bool = False,
    n_votes: int = 1,
    sort_by_length: bool = False,
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
        How many *rows* per call to ``generate_fn``.  Default ``64`` works
        on a 16 GB T4 with a 2B 4-bit model; bump to ``128`` for 24 GB
        cards (L4/A10G) or ``256`` for ≥40 GB (A100/H100).  Set to ``1``
        when wrapping a non-batching API.  Note: when ``n_votes > 1`` the
        actual batch passed to ``generate_fn`` is ``batch_size * n_votes``
        prompts — reduce ``batch_size`` accordingly to stay under GPU
        memory.
    data_dir:
        Override the default local data directory.
    hf_repo:
        HuggingFace repo id to pull data from instead of local files.
        Pass ``"default"`` to resolve against the canonical
        ``tripathysagar/odia-*`` repos.
    reasoning:
        If ``True``, every prompt is wrapped with an Odia instruction
        asking the model to think inside ``<think>...</think>`` and put
        the final answer inside ``\\boxed{...}``.  Gives the model
        scratchpad tokens to compute before committing, and lets the
        scorers pull the answer from a deterministic location.  See
        :func:`odia_eval.build_prompt` for details.
    n_votes:
        Test-time compute via **self-consistency** (Wang et al., 2022).
        ``1`` (default) keeps the legacy one-completion-per-prompt path.
        Values ``>1`` request ``n_votes`` independent completions per
        prompt (the batch is replicated under the hood) and take a
        majority vote over the extracted answers per row.  Requires
        ``generate_fn`` to produce *stochastic* outputs (e.g.
        ``do_sample=True, temperature=0.7``) — otherwise every vote is
        identical and accuracy is unchanged.
    sort_by_length:
        If ``True``, sort prompts within each batch by ``len(prompt)``
        descending before calling ``generate_fn``, then restore the original
        row order before scoring.  With left-padded batches this cuts
        padding/compute overhead when prompt lengths vary; results are
        identical to ``sort_by_length=False`` for a deterministic
        ``generate_fn``.
    verbose:
        Print a progress line after every batch.

    Returns
    -------
    list[EvalRow]
        One entry per evaluated row.
    """
    if benchmark not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}")
    if n_votes < 1:
        raise ValueError(f"n_votes must be >= 1 (got {n_votes})")

    rows = load_benchmark(
        benchmark,
        n_samples=n_samples,
        seed=seed,
        data_dir=data_dir,
        hf_repo=hf_repo,
    )
    total = len(rows)
    results: list[EvalRow] = []

    def _generate_ordered(prompt_list: list[str]) -> list[str]:
        if not sort_by_length or len(prompt_list) <= 1:
            return generate_fn(prompt_list)
        order = sorted(
            range(len(prompt_list)),
            key=lambda i: len(prompt_list[i]),
            reverse=True,
        )
        sorted_prompts = [prompt_list[i] for i in order]
        sorted_preds = generate_fn(sorted_prompts)
        if len(sorted_preds) != len(prompt_list):
            raise ValueError(
                f"generate_fn returned {len(sorted_preds)} outputs for "
                f"{len(prompt_list)} prompts"
            )
        restored: list[str] = [""] * len(prompt_list)
        for sorted_idx, orig_idx in enumerate(order):
            restored[orig_idx] = sorted_preds[sorted_idx]
        return restored

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = [
            build_prompt(benchmark, row, reasoning=reasoning) for row in batch_rows
        ]

        if n_votes == 1:
            predictions = _generate_ordered(prompts)
            if len(predictions) != len(prompts):
                raise ValueError(
                    f"generate_fn returned {len(predictions)} outputs for "
                    f"{len(prompts)} prompts"
                )
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
        else:
            # Replicate each prompt n_votes times so a single generate_fn
            # call covers the whole batch -- preserves any kv-cache /
            # paged-attention batching the user has wired up.
            expanded = [p for p in prompts for _ in range(n_votes)]
            all_preds = _generate_ordered(expanded)
            if len(all_preds) != len(expanded):
                raise ValueError(
                    f"generate_fn returned {len(all_preds)} outputs for "
                    f"{len(expanded)} expanded prompts (batch_size="
                    f"{len(prompts)} x n_votes={n_votes})"
                )
            for i, (row, prompt) in enumerate(zip(batch_rows, prompts)):
                raw_chunk = all_preds[i * n_votes : (i + 1) * n_votes]
                completions = [_strip_prompt(p, prompt) for p in raw_chunk]
                sr = _aggregate_majority_vote(benchmark, completions, row)
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
            vote_tag = f" (n_votes={n_votes})" if n_votes > 1 else ""
            print(
                f"[{benchmark}{vote_tag}] {done}/{total}  "
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
    reasoning: bool = False,
    n_votes: int = 1,
    sort_by_length: bool = False,
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
    reasoning, n_votes, sort_by_length:
        Forwarded to :func:`run_eval` for every benchmark.  See that
        function's docstring for full semantics.  ``reasoning=True``
        gives the model scratchpad time per prompt; ``n_votes>1`` samples
        N independent completions per prompt and majority-votes the
        extracted answer (self-consistency / test-time compute).
        ``sort_by_length=True`` length-buckets each batch before
        generation to reduce left-pad waste.

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
            reasoning=reasoning,
            n_votes=n_votes,
            sort_by_length=sort_by_length,
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
