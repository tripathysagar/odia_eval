"""Quick smoke eval for OdiaBench: random + first-choice baselines.

Validates the eval pipeline end-to-end without a model — useful as a
sanity floor and to confirm `dataset/data/*.jsonl` loads, prompts build,
and scorers extract answers correctly.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odia_eval import BENCHMARKS, full_report, run_all  # noqa: E402

N_SAMPLES = 200
SEED = 42


def make_first_choice_fn():
    """Always answer with the first listed choice for MCQ; '#### 0' for math."""

    def fn(prompts: list[str]) -> list[str]:
        out: list[str] = []
        for p in prompts:
            if "#### N" in p:
                out.append(p + " #### 0")
            elif "(1 ବା 2)" in p:
                out.append(p + " 1")
            else:
                out.append(p + " A")
        return out

    return fn


def make_random_fn(seed: int = 0):
    rng = random.Random(seed)

    def fn(prompts: list[str]) -> list[str]:
        out: list[str] = []
        for p in prompts:
            if "#### N" in p:
                out.append(p + f" #### {rng.randint(0, 100)}")
            elif "(1 ବା 2)" in p:
                out.append(p + " " + rng.choice(["1", "2"]))
            elif "A, B, C ବା D" in p or "A, B, C, D" in p:
                out.append(p + " " + rng.choice(["A", "B", "C", "D"]))
            else:
                # truthfulqa: count labels in the prompt
                import re

                labels = re.findall(r"^([A-M]):", p, flags=re.MULTILINE)
                pool = labels or ["A"]
                out.append(p + " " + rng.choice(pool))
        return out

    return fn


def main() -> None:
    print("Benchmarks:", BENCHMARKS, flush=True)

    for label, fn in [
        ("first-choice", make_first_choice_fn()),
        ("random",       make_random_fn(seed=0)),
    ]:
        print(f"\n=== baseline: {label} (n_samples={N_SAMPLES}) ===", flush=True)
        t0 = time.time()
        results = run_all(fn, n_samples=N_SAMPLES, seed=SEED, batch_size=64, verbose=False)
        dt = time.time() - t0
        print(full_report(results))
        print(f"({dt:.1f}s)")


if __name__ == "__main__":
    main()
