"""Dataset loaders for OdiaBench evaluation.

Each loader returns a list of plain dicts from the *test / validation* split
only (no training data is ever loaded).  Pass ``n_samples`` for a random
reproducible subset; omit it (or pass ``None``) to use the full split.

Default data source: ``dataset/data/`` inside the OdiaBench repo.
Override with ``data_dir`` to point at any directory that contains the same
``.jsonl`` files, or set ``hf_repo`` to pull directly from a HuggingFace Hub
dataset (requires ``datasets`` installed and HF_TOKEN if private).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

# Canonical benchmark names and the file stem they map to in dataset/data/.
# Files are prefixed ``odia-`` to mirror the HuggingFace Hub repo names
# under ``tripathysagar/odia-{benchmark}``.
_BENCHMARK_STEMS: dict[str, str] = {
    "gsm8k":      "odia-gsm8k_test",
    "arc":        "odia-arc_validation",
    "truthfulqa": "odia-truthfulqa_mc_validation",
    "winogrande": "odia-winogrande_validation",
    "hellaswag":  "odia-hellaswag_validation",
}

# Canonical HuggingFace Hub repos for each benchmark.  Pass
# ``hf_repo="default"`` to ``load_benchmark`` / ``load_all`` to resolve
# against this map instead of the local files.
_HF_REPO_OWNER = "tripathysagar"
DEFAULT_HF_REPOS: dict[str, str] = {
    "gsm8k":      f"{_HF_REPO_OWNER}/odia-gsm8k",
    "arc":        f"{_HF_REPO_OWNER}/odia-arc",
    "truthfulqa": f"{_HF_REPO_OWNER}/odia-truthfulqa-mc",
    "winogrande": f"{_HF_REPO_OWNER}/odia-winogrande",
    "hellaswag":  f"{_HF_REPO_OWNER}/odia-hellaswag",
}

BENCHMARKS: list[str] = sorted(_BENCHMARK_STEMS)

# Repo root is two levels up from this file (odia_eval/datasets.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "dataset" / "data"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_benchmark(
    name: str,
    *,
    n_samples: int | None = None,
    seed: int = 42,
    data_dir: str | Path | None = None,
    hf_repo: str | None = None,
) -> list[dict[str, Any]]:
    """Load one OdiaBench benchmark split.

    Parameters
    ----------
    name:
        Benchmark key — one of ``"gsm8k"``, ``"arc"``, ``"truthfulqa"``,
        ``"winogrande"``, ``"hellaswag"``.
    n_samples:
        How many rows to return.  ``None`` returns all rows.  When provided,
        rows are drawn uniformly at random (without replacement) using
        ``seed`` so results are reproducible.
    seed:
        Random seed for sampling.  Ignored when ``n_samples`` is ``None``.
    data_dir:
        Directory that contains ``<stem>.jsonl`` files.  Defaults to
        ``dataset/data/`` in the OdiaBench repo root.
    hf_repo:
        HuggingFace dataset id (e.g. ``"tripathysagar/odia-gsm8k"``).
        When set, the data is pulled from the Hub instead of the local files.
        Pass the sentinel string ``"default"`` to resolve against
        :data:`DEFAULT_HF_REPOS` (the canonical ``tripathysagar/odia-*``
        repos).  Requires ``datasets`` to be installed.

    Returns
    -------
    list[dict]
        Each dict has at minimum ``id``, ``question``, ``answer``,
        ``odia_question``, ``odia_answer``, plus benchmark-specific extras.
    """
    if name not in _BENCHMARK_STEMS:
        raise KeyError(
            f"unknown benchmark {name!r}; expected one of {BENCHMARKS}"
        )

    if hf_repo == "default":
        hf_repo = DEFAULT_HF_REPOS[name]

    if hf_repo is not None:
        rows = _load_from_hub(hf_repo, name)
    else:
        stem = _BENCHMARK_STEMS[name]
        base = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        path = base / f"{stem}.jsonl"
        if not path.exists():
            # Auto-fallback: try the Hub before raising
            try:
                rows = _load_from_hub(DEFAULT_HF_REPOS[name], name)
            except Exception:
                raise FileNotFoundError(
                    f"Dataset file not found: {path}\n"
                    f"Run  python -m odia_trans --dataset "
                    f"{stem.removeprefix('odia-')} merge  "
                    f"inside cursor_translate/ to generate it, or pass "
                    f"hf_repo='default' to pull from "
                    f"{DEFAULT_HF_REPOS[name]}."
                )
        else:
            rows = _load_jsonl(path)

    return _sample(rows, n_samples, seed)


def _load_from_hub(repo: str, name: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "Install 'datasets' to use hf_repo loading: pip install datasets"
        ) from exc

    split_map = {
        "gsm8k":      "test",
        "arc":        "validation",
        "truthfulqa": "validation",
        "winogrande": "validation",
        "hellaswag":  "validation",
    }
    ds = hf_load(repo, split=split_map[name])
    return [dict(row) for row in ds]


def load_all(
    *,
    n_samples: int | None = None,
    seed: int = 42,
    data_dir: str | Path | None = None,
    hf_repos: dict[str, str] | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load all five benchmarks at once.

    Parameters
    ----------
    hf_repos:
        * ``None`` (default) — load every benchmark from local files.
        * ``"default"`` — load every benchmark from :data:`DEFAULT_HF_REPOS`
          (the canonical ``tripathysagar/odia-*`` repos).
        * ``dict`` — mapping of ``benchmark_name → hf_repo_id`` for any
          benchmarks you want to load from HF Hub.  Benchmarks absent from
          this dict fall back to local files.
    """
    if hf_repos == "default":
        hf_repos = dict(DEFAULT_HF_REPOS)
    hf_repos = hf_repos or {}
    return {
        name: load_benchmark(
            name,
            n_samples=n_samples,
            seed=seed,
            data_dir=data_dir,
            hf_repo=hf_repos.get(name),
        )
        for name in BENCHMARKS
    }
