# OdiaBench Results

Per-model scoreboard for the five translated OdiaBench reasoning benchmarks. Each row reports the percentage of correctly extracted answers on the full test/validation split, the 95% Wilson CI, and the gap above the per-task chance baseline.

> Scores are produced by `odia_eval` (≥ 0.4.0) — see [`notebooks/eval_odiabench_working.ipynb`](./notebooks/eval_odiabench_working.ipynb) for the full reference pipeline.

## `google/gemma-4-E2B-it`

Gemma 4 2B IT, 4-bit NF4 via BitsAndBytes. Full splits, greedy decoding, no self-consistency.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  502 / 1172     | 42.8 %   | 40.0 – 45.7   | 25.1 % | **+17.7** |
| arc / validation         |  120 /  299     | 40.1 %   | 34.7 – 45.8   | 25.1 % | **+15.0** |
| gsm8k / test             |  345 / 1319     | 26.2 %   | 23.9 – 28.6   |  0.0 % | **+26.2** |
| hellaswag / validation   | 4183 / 10042    | 41.7 %   | 40.7 – 42.6   | 25.0 % | **+16.7** |
| truthfulqa / validation  |  360 /  817     | 44.1 %   | 40.7 – 47.5   | 22.6 % | **+21.5** |
| winogrande / validation  |  654 / 1267     | 51.6 %   | 48.9 – 54.4   | 50.0 % |   +1.6   |
| **MEAN (unweighted)**    |                 | **41.1 %** |             |        | **+16.4** |

### Run metadata

| Key | Value |
|---|---|
| `odia_eval` version       | 0.4.0 |
| Quantisation              | NF4 4-bit (BitsAndBytes, double-quant, fp16 compute) |
| `attn_implementation`     | auto → flash_attention_2 / sdpa fallback |
| `torch.compile`           | enabled, `mode="default"` (post-fix; reduce-overhead also works via `cudagraph_mark_step_begin`) |
| `sort_by_length`          | False |
| `reasoning`               | False (no `<think>` / `\boxed{}` wrapping) |
| `n_votes`                 | 1 (greedy, no self-consistency) |
| `BATCH_SIZE`              | 512 for MCQ, 128 for GSM8K |
| Seed                      | 42 |
| Data source               | HuggingFace Hub `tripathysagar/odia-*` (parquet shards) |

### Failure-mode notes

* **Winogrande (51.6 %, +1.6 over chance).** Essentially at random. Every spot-checked failure picks option 2; in several cases the model writes the literal *content* of option 2 (`"...**2: Hunter**"`) suggesting it resolves the coreference correctly but the gold label points at option 1. Two plausible explanations:
  * The Odia translation flipped `option1` / `option2` ordering relative to `answer_idx`.
  * The Odia sentence dropped the disambiguating context that pins the referent.

  Spot-check before treating this as a model weakness:

  ```python
  from odia_eval import show_failures
  show_failures(all_results["winogrande/validation"], n=10, max_chars=400)
  ```

* **HellaSwag (41.7 %).** The expected ceiling for "Odia context + English endings". Failure rows literally echo the English ending text (`'D: stands on his hands and springs.'`). Translating the four endings (currently only the gold ending is translated) is the V2 fix flagged in the package README.

* **GSM8K (26.2 %).** Reasonable for a 2B model on Odia chain-of-thought. Failures span numeric off-by-orders (`120000` vs `70000`), early-truncation of the reasoning, and the model committing to a partial step (`"... 2."`) before reaching the final number. `n_votes>1` with `reasoning=True` and a stochastic generate_fn would likely move this 3–6 points.

* **ARC, TruthfulQA.** Both comfortably above chance with tight CIs. ARC's test split (1172 rows) is the most statistically reliable single number in the table.

### Known runtime caveats during this run

* Dynamo hit `config.recompile_limit (8)` partway through `arc/test`, falling back to eager for the remainder of that split. Bump `torch._dynamo.config.recompile_limit = 64` in the compile cell to avoid this on subsequent runs.
* Hundreds of `cudagraph partition due to non gpu ops` lines were silenced in 0.4.0 by raising the `torch._inductor.cudagraph_trees` logger to `ERROR`.
