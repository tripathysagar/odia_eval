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

## Reasoning prompts (`<think>` + `\boxed{}`)

Pass `reasoning=True` to give the model scratchpad tokens before it commits to an answer. Every prompt is wrapped with an Odia instruction asking it to think inside `<think>...</think>` and put the final answer inside `\boxed{...}`. The scorers already prefer `\boxed{...}` content when present, so no scoring change is needed:

```python
# Single benchmark with reasoning prompts
results = run_eval("gsm8k", my_generate, n_samples=200, reasoning=True)

# All five benchmarks with reasoning prompts
all_results = run_all(my_generate, n_samples=200, reasoning=True)
```

The reasoning prefix (verbatim):

> ତୁମେ ପ୍ରଥମେ `<think>` ଓ `</think>` ଭିତରେ ଧାପ ଧାପ ଭାବରେ ଚିନ୍ତା କର, ଶେଷରେ କେବଳ ଚୂଡାନ୍ତ ଉତ୍ତରଟି `\boxed{...}` ଭିତରେ ଲେଖ।

Bump `max_new_tokens` (e.g. to 512 or 1024) when `reasoning=True` so the model has room to think before reaching the box. The extractor pulls the **last** `\boxed{...}` in the output, so any scratch boxes earlier in the reasoning are ignored.

## Test-time compute (`n_votes`, self-consistency)

Pass `n_votes>1` to sample N independent completions per prompt and **majority-vote** the extracted answer per row. This is the Wang et al. (2022) self-consistency recipe — typically the biggest single-knob accuracy bump on math / MCQ tasks at inference time:

```python
# Each prompt sampled 5 times; majority vote chooses the final answer.
# Requires generate_fn to be STOCHASTIC (do_sample=True).
results = run_eval(
    "gsm8k",
    my_generate,
    n_samples=200,
    reasoning=True,
    n_votes=5,
)
```

Notes:

- `generate_fn` must be **stochastic** — e.g. `model.generate(..., do_sample=True, temperature=0.7, top_p=0.95)`. Greedy decoding produces identical votes and gains nothing.
- The actual batch passed to `generate_fn` is `batch_size * n_votes`. Drop `batch_size` accordingly if you hit OOM (e.g. `batch_size=16, n_votes=8` ≈ 128 prompts per call).
- Votes are canonicalised before counting: `"1.5"` and `"1.50"` count as the same vote on GSM8K; `"a"` and `"A"` count as the same vote on MCQ tasks.
- Empty extractions are treated as abstentions and never win the vote.
- Ties are broken in first-seen order.

Pairs naturally with `reasoning=True`: the reasoning prompt produces a `\boxed{}` per sample, the scorer extracts each, the voter picks the consensus.

## API reference

### `run_eval(benchmark, generate_fn, *, n_samples, seed, batch_size, reasoning=False, n_votes=1, sort_by_length=False, ...)`
Evaluate one benchmark. Returns `list[EvalRow]`. Set `reasoning=True` for `<think>`+`\boxed{}` prompts, `n_votes>1` for self-consistency majority voting, and `sort_by_length=True` to length-bucket each batch before generation.

### `run_all(generate_fn, *, n_samples, seed, batch_size, reasoning=False, n_votes=1, sort_by_length=False, skip, ...)`
Evaluate all five benchmarks. Returns `dict[str, list[EvalRow]]`. Same `reasoning`, `n_votes`, and `sort_by_length` knobs as `run_eval`.

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

### `build_prompt(benchmark, row, *, reasoning=False)` → `str`
Build the Odia-language prompt for one row. Pass `reasoning=True` to prepend the `<think>`+`\boxed{}` reasoning instruction.

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

## Faster inference with `torch.compile`

Compile only the forward pass (not `model.generate`, which has Python control flow that doesn't play well with Dynamo). `dynamic=True` makes Inductor capture a single graph that handles every batch / prompt-length shape, so it never recompiles on the last (partial) batch.

```python
import torch

# Apply once, right after model.eval()
if torch.cuda.is_available() and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 4):
    try:
        model.forward = torch.compile(
            model.forward,
            mode="reduce-overhead",   # CUDA graphs — best for repeated forwards
            dynamic=True,             # one graph handles all (B, T) shapes
            fullgraph=False,          # tolerate small graph breaks
        )
        print("torch.compile applied")
    except Exception as e:
        print(f"[warn] torch.compile failed ({type(e).__name__}: {e}); using eager")
```

Notes:

- First forward pass is slow (~30–120 s graph capture). The eval loop's progress bar will sit on the first batch — that's expected; every subsequent batch runs on the compiled graph.
- Typical gains: **1.3–2x on Ampere+** (A100/L4/H100) for generation-heavy benchmarks like GSM8K; smaller on Turing (T4). Pairs especially well with `n_votes>1` — more forward passes per row means the compile cost is amortised over more work.
- Requires `torch>=2.4` (2.5+ recommended). On older torch or older `bitsandbytes`, `torch.compile` over 4-bit BnB layers can fail; the try/except above falls back to eager so the eval still runs.
- The `eval_odiabench.ipynb` and `eval_odiabench_working.ipynb` notebooks both expose this via `USE_TORCH_COMPILE = True` and `TORCH_COMPILE_MODE = "reduce-overhead"` in the config cell.

## Maximum-throughput inference with vLLM

For large batched evals, wrapping `vllm.LLM.generate` in your `generate_fn` typically yields **5–10× throughput** vs HuggingFace `model.generate` on the same GPU — vLLM's PagedAttention and continuous batching amortise KV-cache memory and keep the GPU fed.

Trade-offs:

- Separate dependency (`pip install vllm`) and a slower cold start while weights load into vLLM's engine.
- Harder to pair with BitsAndBytes 4-bit in-process quantisation — vLLM expects AWQ/GPTQ checkpoints or full-precision weights loaded through its own path.
- You still slice completions off the prompt before scoring (or rely on `run_eval`'s `_strip_prompt` safety net).

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-2-2b-it",   # or your Odia SFT checkpoint
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
)

def make_vllm_generate_fn(*, do_sample: bool = False, n: int = 1, temperature: float = 0.7):
    """Return a generate_fn compatible with run_eval / run_all."""

    def _generate(prompts: list[str]) -> list[str]:
        if do_sample:
            params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=256,
                n=n,
            )
        else:
            params = SamplingParams(max_tokens=256, temperature=0.0)

        outputs = llm.generate(prompts, params)
        # vLLM returns one RequestOutput per prompt; flatten when n>1.
        decoded: list[str] = []
        for out in outputs:
            for cand in out.outputs:
                decoded.append(cand.text)
        if not do_sample and n == 1:
            return decoded
        # Self-consistency: run_eval replicates each prompt n_votes times, so
        # keep greedy n=1 and let run_eval handle replication instead.
        return decoded

    return _generate

generate_fn = make_vllm_generate_fn(do_sample=False)
results = run_eval("arc", generate_fn, n_samples=200, batch_size=128)

# Self-consistency — stochastic generate_fn + n_votes in run_eval
sample_fn = make_vllm_generate_fn(do_sample=True, temperature=0.7)
results = run_eval("gsm8k", sample_fn, n_samples=200, reasoning=True, n_votes=5, batch_size=32)
```

## Inference optimizations (notebooks)

The reference eval notebooks (`eval_odiabench.ipynb`, `eval_odiabench_working.ipynb`) expose several optional knobs in the config cell:

| Knob | Default | Effect |
|---|---|---|
| `ATTN_IMPL` | `"auto"` | `"auto"` tries `flash_attention_2` → `sdpa` → default; pass an explicit string to force one backend |
| `USE_TORCH_COMPILE` | `True` | Compiles `model.forward` via TorchDynamo (see above) |
| `USE_STATIC_CACHE` | `False` | With compile enabled, sets `generation_config.cache_implementation="static"` and recompiles with `fullgraph=True, dynamic=False` for another 2–3× on Ampere+ |
| `SORT_BY_LENGTH` | `False` | Forwards to `run_eval(..., sort_by_length=True)` — sorts each batch by prompt length (descending) before generation to cut left-pad waste |
| MCQ early stop | on for `generate_mcq` | Stopping criteria halt once every row in the batch closes `\boxed{...}` or hits a blank line — saves tokens vs `max_new_tokens=32` |

Programmatic use:

```python
results = run_eval("arc", my_generate, n_samples=200, sort_by_length=True)
```

Run metadata sidecars (`.meta.json` next to each results file) record `attn_implementation`, `use_static_cache`, `early_stop_on_box`, and `sort_by_length` for reproducibility.

### Known limitations

- **`USE_STATIC_CACHE` + 4-bit BnB** — static KV-cache with `fullgraph=True` compile often fails on BitsAndBytes-quantised models because batch shapes and cache layout vary. The notebooks detect this and skip the static path with a printed warning; leave `USE_STATIC_CACHE=False` (default) on T4 / BnB runs.
- **FlashAttention-2** — requires the `flash-attn` package and Ampere+ (`sm_80+`). When unavailable the notebooks silently fall back through SDPA to eager attention.

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
| **Prompts** | One Odia-language prompt builder per benchmark; handles MCQ blocks, fill-in-the-blank, chain-of-thought math, cross-lingual choices; optional `reasoning=True` wraps every prompt with a `<think>...</think>` + `\boxed{...}` instruction so models get scratchpad tokens before committing |
| **Scoring** | Post-generation regex extraction — no perplexity needed; `\boxed{...}` content preferred when present (reasoning path), with fallback to: `#### N` for GSM8K, A–E for ARC, A–M for TruthfulQA (choices shuffled deterministically by row id so gold isn't always "A"), A–D for HellaSwag, 1/2 for Winogrande (Odia digits ୦-୯ and number-words ଗୋଟିଏ/ଦୁଇ normalised before extraction); `run_eval` automatically strips the prompt prefix from each prediction so prompt-echoing `generate_fn`s don't bias the regex |
| **Eval loop** | `run_eval` / `run_all` — model-agnostic (takes any `generate_fn`); live batch progress; per-benchmark batch size control; `sort_by_length` length-buckets batches; `n_votes>1` enables self-consistency majority voting (test-time compute) on top of any stochastic `generate_fn` |
| **Reporting** | `score_report`, `full_report` (95% Wilson CI + per-task chance baseline + Δ-above-chance), `to_records` (pandas export), `save_results` / `load_results` (JSONL round-trip), `show_failures` (debug helper) |

### Boundary

```
odia_trans              odia_eval                notebooks/eval_odiabench.ipynb
──────────────    ──────────────────────    ────────────────────────────────
translate data  → load rows, build         load model (4-bit NF4),
into JSONL        prompts, score outputs,  define generate_fn,
                  report accuracy           save results to JSONL
```

Three runnable notebooks live under [`notebooks/`](./notebooks):

* [`capability.ipynb`](./notebooks/capability.ipynb) — pure-CPU walkthrough of the `odia_eval` API (no model required). [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/capability.ipynb)
* [`eval_odiabench.ipynb`](./notebooks/eval_odiabench.ipynb) — full 4-bit Gemma reference eval on Kaggle T4 / Colab GPU (loads benchmark data from a cloned OdiaBench repo). [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/eval_odiabench.ipynb)
* [`eval_odiabench_working.ipynb`](./notebooks/eval_odiabench_working.ipynb) — same eval but data is streamed from the canonical `tripathysagar/odia-*` HF Hub repos, with multi-split support (`arc/test` + `arc/validation`, etc.). [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/eval_odiabench_working.ipynb)

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
