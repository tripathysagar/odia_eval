"""Statistical helpers and result-inspection utilities.

Adds Wilson-score confidence intervals (more reliable than the normal
approximation at small ``n`` and near 0/1), known chance baselines per
benchmark, and a ``show_failures`` debugging helper.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .eval import EvalRow


# Chance accuracy for a uniformly random output on each benchmark.
# Computed from the actual local data (see notebooks/capability.ipynb for
# the recomputation snippet):
#   gsm8k      → free-form numeric answer, effectively 0
#   arc        → 295 rows × 4 choices, 3 × 3 choices, 1 × 5 choices
#   truthfulqa → mean(1/n_choices) over 817 rows (n varies from 4 to 13)
#   winogrande → strict 1/2
#   hellaswag  → strict 1/4
CHANCE_LEVELS: dict[str, float] = {
    "gsm8k":      0.000,
    "arc":        0.251,
    "truthfulqa": 0.226,
    "winogrande": 0.500,
    "hellaswag":  0.250,
}


def chance_level(benchmark: str) -> float:
    """Return the theoretical random-baseline accuracy for ``benchmark``."""
    return CHANCE_LEVELS.get(benchmark, float("nan"))


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion ``k/n``.

    Returns ``(lo, hi)`` as proportions in ``[0, 1]``.  The Wilson
    interval is preferred over the normal (Wald) approximation because
    it stays inside ``[0, 1]`` even when ``k == 0`` or ``k == n`` and is
    well-calibrated for small samples (which is exactly the regime
    OdiaBench tends to run in — ``n_samples=200`` per benchmark).
    """
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def accuracy_with_ci(
    results: Iterable["EvalRow"],
    alpha: float = 0.05,
) -> tuple[float, float, float, int, int]:
    """Return ``(accuracy, ci_lo, ci_hi, correct, total)`` for one result list."""
    results = list(results)
    total = len(results)
    correct = sum(r.correct for r in results)
    acc = correct / total if total else 0.0
    lo, hi = wilson_ci(correct, total, alpha=alpha)
    return acc, lo, hi, correct, total


def show_failures(
    results: Iterable["EvalRow"],
    n: int = 10,
    max_chars: int = 200,
) -> None:
    """Print up to ``n`` incorrectly scored rows for failure-mode inspection.

    Shows ``prompt`` (truncated), ``gold``, ``extracted``, and the raw
    ``prediction`` (truncated) — exactly the four fields you need to
    decide whether the model was wrong or the extractor was wrong.
    """
    fails = [r for r in results if not r.correct][:n]
    if not fails:
        print("(no failures to show)")
        return

    def _trunc(s: str) -> str:
        s = str(s).replace("\n", " ")
        return s if len(s) <= max_chars else s[:max_chars] + "…"

    for r in fails:
        print(f"── {r.benchmark} #{r.row_id} ──")
        print(f"  prompt:     {_trunc(r.prompt)}")
        print(f"  gold:       {r.score_result.gold!r}")
        print(f"  extracted:  {r.score_result.extracted!r}")
        print(f"  prediction: {_trunc(r.score_result.prediction)}")
        print()
