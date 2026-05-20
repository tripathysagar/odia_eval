"""Persist and reload ``run_eval`` / ``run_all`` results as JSONL.

Saving is one line per evaluated row (benchmark, id, prompt, gold,
extracted, prediction, correct).  Loading reconstructs the original
``dict[str, list[EvalRow]]`` shape so all downstream helpers
(``score_report``, ``full_report``, ``show_failures``, ``to_records``)
keep working without re-running generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .eval import EvalRow
from .scorers import ScoreResult


def save_results(
    all_results: dict[str, list[EvalRow]] | Iterable[EvalRow],
    path: str | Path,
) -> int:
    """Write evaluation results to a JSONL file.

    Accepts either the ``dict`` returned by :func:`run_all` or the
    ``list`` returned by :func:`run_eval`.  Returns the number of rows
    written.  Files are UTF-8 with ``ensure_ascii=False`` so Odia text
    is human-readable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(all_results, dict):
        rows: list[EvalRow] = [r for bench_rows in all_results.values() for r in bench_rows]
    else:
        rows = list(all_results)

    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            rec = {
                "benchmark":  r.benchmark,
                "id":         r.row_id,
                "prompt":     r.prompt,
                "correct":    r.correct,
                "extracted":  r.score_result.extracted,
                "gold":       r.score_result.gold,
                "prediction": r.score_result.prediction,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(rows)


def load_results(path: str | Path) -> dict[str, list[EvalRow]]:
    """Load a JSONL file produced by :func:`save_results`.

    Returns the same ``dict[str, list[EvalRow]]`` shape as
    :func:`run_all`, with benchmarks ordered as they first appear in
    the file.
    """
    path = Path(path)
    out: dict[str, list[EvalRow]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sr = ScoreResult(
                correct=rec["correct"],
                extracted=rec["extracted"],
                gold=rec["gold"],
                prediction=rec["prediction"],
            )
            row = EvalRow(
                benchmark=rec["benchmark"],
                row_id=rec["id"],
                prompt=rec.get("prompt", ""),
                score_result=sr,
            )
            out.setdefault(rec["benchmark"], []).append(row)
    return out
