# odia_eval

Evaluation utilities for [OdiaBench](../README.md) — the community-curated Odia LLM benchmark suite.

## Install (Kaggle / Colab)

```python
# Step 1 — clone the repo (once per session)
!git clone https://github.com/tripathysagar/OdiaBench

# Step 2 — add to path
import sys
sys.path.insert(0, "/kaggle/working/OdiaBench")   # Kaggle
# sys.path.insert(0, "/content/OdiaBench")        # Colab
```

No pip install required.

## Quick start

```python
from odia_eval import run_eval, run_all, score_report, full_report

# --- define your generate function ---
# Receives: list[str] of prompts
# Returns:  list[str] of decoded model outputs (same length)

def my_generate(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# --- single benchmark, 200 random samples ---
results = run_eval("gsm8k", my_generate, n_samples=200, seed=42)
print(score_report(results))
# gsm8k         |  correct:  42/200  |  accuracy: 21.0%

# --- all five benchmarks (skip hellaswag for speed) ---
all_results = run_all(my_generate, n_samples=200, skip=["hellaswag"])
print(full_report(all_results))
```

## Benchmarks

| Name | Local file | HF Hub repo | Samples | Task | Scoring |
|---|---|---|---|---|---|
| `gsm8k` | `odia-gsm8k_test.jsonl` | `tripathysagar/odia-gsm8k` | 1,319 | Math reasoning | Extract `#### N`, exact match |
| `arc` | `odia-arc_validation.jsonl` | `tripathysagar/odia-arc` | 299 | Science MCQ | Extract A–E, exact match |
| `truthfulqa` | `odia-truthfulqa_mc_validation.jsonl` | `tripathysagar/odia-truthfulqa-mc` | 817 | Truthfulness MC | Extract A–M (choices shuffled by row id), exact match |
| `winogrande` | `odia-winogrande_validation.jsonl` | `tripathysagar/odia-winogrande` | 1,267 | Commonsense fill-in | Extract 1/2 (Odia digits and ଗୋଟିଏ/ଦୁଇ accepted), exact match |
| `hellaswag` | `odia-hellaswag_validation.jsonl` | `tripathysagar/odia-hellaswag` | 10,042 | Sentence completion | Extract A/B/C/D, exact match |

> All benchmarks use **test / validation splits only** — no training rows are ever loaded.

## Random sampling

```python
# Reproducible 100-row sample
results = run_eval("arc", my_generate, n_samples=100, seed=0)

# Full split (no sampling)
results = run_eval("arc", my_generate)
```

The same `seed` always yields the same rows, so results are comparable across models.

## API reference

### `run_eval(benchmark, generate_fn, *, n_samples, seed, batch_size, ...)`
Evaluate one benchmark. Returns `list[EvalRow]`.

### `run_all(generate_fn, *, n_samples, seed, batch_size, skip, ...)`
Evaluate all five benchmarks. Returns `dict[str, list[EvalRow]]`.

### `score_report(results, alpha=0.05)` → `str`
One-line accuracy summary with **95% Wilson confidence interval** and the gap above the known chance baseline.

### `full_report(all_results, alpha=0.05)` → `str`
Multi-line accuracy table for all benchmarks, with Wilson CIs, per-task chance baselines (`CHANCE_LEVELS`), and a mean row.

```
OdiaBench Evaluation Results  (95% Wilson CI)
============================================================
arc            61/200    30.5%  [ 24.6– 37.1]  chance 25.1%  Δ  +5.4
gsm8k          42/200    21.0%  [ 15.9– 27.1]  chance  0.0%  Δ +21.0
hellaswag      55/200    27.5%  [ 21.7– 34.1]  chance 25.0%  Δ  +2.5
truthfulqa     88/200    44.0%  [ 37.3– 50.9]  chance 22.6%  Δ +21.4
winogrande     99/200    49.5%  [ 42.6– 56.4]  chance 50.0%  Δ  -0.5
------------------------------------------------------------
MEAN                     34.5%                                 Δ  +9.9
```

### `wilson_ci(k, n, alpha=0.05)` → `(lo, hi)`
Wilson score interval for a binomial proportion — well-calibrated at small `n` and stays in `[0, 1]` near the boundaries.

### `accuracy_with_ci(results, alpha=0.05)` → `(acc, lo, hi, correct, total)`
The numbers behind `score_report`, if you want to format your own table.

### `chance_level(benchmark)` / `CHANCE_LEVELS`
Theoretical random-baseline accuracy (`gsm8k 0.0`, `arc 0.251`, `truthfulqa 0.226`, `winogrande 0.5`, `hellaswag 0.25`). Computed from the actual local data — TFQA's value averages `1/n_choices` over the 817 rows.

### `show_failures(results, n=10, max_chars=200)`
Prints prompt + gold + extracted + raw prediction for the first N wrong rows. The quickest way to decide whether the model was wrong or the extractor was wrong.

### `save_results(all_results, path)` / `load_results(path)`
JSONL round-trip for `run_eval` / `run_all` output. Re-score later (after fixing an extractor, tweaking the prompt, or adding a new metric) without re-running expensive generation. Writes UTF-8 with Odia text human-readable.

### `to_records(all_results)` → `list[dict]`
Flatten results to a list of dicts — convenient for pandas or jsonl export.

```python
import pandas as pd
df = pd.DataFrame(to_records(all_results))
df.to_csv("results.csv", index=False)
```

### `load_benchmark(name, *, n_samples, seed, data_dir, hf_repo)`
Load raw rows directly (useful for custom eval loops).

### `build_prompt(benchmark, row)` → `str`
Build the Odia-language prompt for one row.

### `score(benchmark, prediction, row)` → `ScoreResult`
Score one prediction. Returns `ScoreResult(correct, extracted, gold, prediction)`.

## 4-bit model setup (recommended for Kaggle T4)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"    # Gemma 4 2B IT

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
)
model.eval()

def generate_fn(prompts):
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False)
    # slice off the prompt tokens
    gen_ids = [o[len(i):] for o, i in zip(out, enc["input_ids"])]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
```

## Data directory

By default `odia_eval` looks for `dataset/data/odia-*.jsonl` inside the
cloned OdiaBench repo. If you store the parquet/jsonl files elsewhere:

```python
results = run_eval("gsm8k", fn, data_dir="/my/custom/path")
```

## Load from HuggingFace Hub

The canonical Hub repos live under `tripathysagar/odia-*`. Pass
`hf_repo="default"` (single benchmark) or `hf_repos="default"` (every
benchmark) to bypass the local files:

```python
# Single benchmark from the Hub
results = run_eval("gsm8k", fn, n_samples=200, hf_repo="default")

# All five benchmarks from the Hub
all_results = run_all(fn, n_samples=200, hf_repos="default")

# Or mix-and-match: only hellaswag from the Hub, the rest local
all_results = run_all(
    fn,
    hf_repos={"hellaswag": "tripathysagar/odia-hellaswag"},
)
```

Inspect the default mapping via `from odia_eval import DEFAULT_HF_REPOS`.

---

## Scope

### In scope

| Area | What `odia_eval` does |
|---|---|
| **Datasets** | Loads the 5 test/validation splits; reproducible random sampling via `n_samples` + `seed`; HF Hub fallback via `hf_repo=` |
| **Prompts** | One Odia-language prompt builder per benchmark; handles MCQ blocks, fill-in-the-blank, chain-of-thought math, cross-lingual choices |
| **Scoring** | Post-generation regex extraction — no perplexity needed; `#### N` for GSM8K, A–E for ARC, A–M for TruthfulQA (choices shuffled deterministically by row id so gold isn't always "A"), A–D for HellaSwag, 1/2 for Winogrande (Odia digits ୦-୯ and number-words ଗୋଟିଏ/ଦୁଇ normalised before extraction); `run_eval` automatically strips the prompt prefix from each prediction so prompt-echoing `generate_fn`s don't bias the regex |
| **Eval loop** | `run_eval` / `run_all` — model-agnostic (takes any `generate_fn`); live batch progress; per-benchmark batch size control |
| **Reporting** | `score_report`, `full_report` (95% Wilson CI + per-task chance baseline + Δ-above-chance), `to_records` (pandas export), `save_results` / `load_results` (JSONL round-trip), `show_failures` (debug helper) |

### Boundary

```
odia_trans              odia_eval                notebooks/eval_odiabench.ipynb
──────────────    ──────────────────────    ────────────────────────────────
translate data  → load rows, build         load model (4-bit NF4),
into JSONL        prompts, score outputs,  define generate_fn,
                  report accuracy           save results to JSONL
```

Two runnable notebooks live under [`notebooks/`](../notebooks/):

* [`notebooks/capability.ipynb`](../notebooks/capability.ipynb) — pure-CPU walkthrough of the `odia_eval` API (no model required). [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/OdiaBench/blob/main/notebooks/capability.ipynb)
* [`notebooks/eval_odiabench.ipynb`](../notebooks/eval_odiabench.ipynb) — full 4-bit Gemma reference eval on Kaggle T4 / Colab GPU. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/OdiaBench/blob/main/notebooks/eval_odiabench.ipynb)

`odia_eval` owns everything **after** the data exists and **before** the model is defined.  
It knows nothing about how the model was built, quantised, or trained.

### Explicitly out of scope

| Area | Lives elsewhere |
|---|---|
| Translation pipeline (sharding, manifests, QE fixers) | `cursor_translate/odia_trans/` |
| Model loading, 4-bit quantisation, chat templates | `notebooks/eval_odiabench.ipynb` |
| Training / SFT / DPO data loading | Phase 2 work (separate pipeline) |
| Register-only Indic evals (MILU, IndicIFEval, IndicGenBench, FLORES+) | Not yet integrated — Phase L4 in `odia_trans` agenda |
| Back-translation QE (NLLB cosine, faithfulness) | `odiabench_quality_eval.ipynb` |
| Likelihood-based MCQ scoring (log-prob over choices) | V2 scope — see gaps below |
| Multi-language eval (Bengali, Bodo, Hindi) | V2, after paper |
| HF leaderboard publish / dataset cards | Separate publish step |

### Known gaps (V2 candidates)

1. **Likelihood scoring** — for TruthfulQA and Winogrande, perplexity over choices is more reliable than letter extraction from greedy output. V2 adds `score_logprob(benchmark, logprobs, row)`.

2. **Register-only Indic evals** — MILU, IndicIFEval, IndicGenBench, FLORES+ are in the meta agenda but not in `BENCHMARKS` yet. Requires new loaders + prompt builders once those datasets are registered in `odia_trans`.

3. **HellaSwag distractors not in Odia** — `all_endings` are English. The model sees an Odia context + English endings. A future mapper could translate all 4 endings, making the task fully Odia.

4. **No CLI** — currently library-only. A `python -m odia_eval --benchmark gsm8k --model ...` interface would make it usable outside notebooks.
