# OdiaBench Results

Per-model scoreboard for the five translated OdiaBench reasoning benchmarks. Each row reports the percentage of correctly extracted answers on the full test/validation split, the 95% Wilson CI, and the gap above the per-task chance baseline.

> Scores are produced by `odia_eval` (≥ 0.4.0) — see [`notebooks/eval_odiabench_working.ipynb`](./notebooks/eval_odiabench_working.ipynb) for the full reference pipeline. Per-model notebooks with cached outputs live under [`notebooks/`](./notebooks).

## Summary

Models sorted by unweighted MEAN accuracy across the splits actually evaluated.

| Model | Quant | arc/test | gsm8k/test | hellaswag | truthfulqa | winogrande | MEAN | Δ vs chance |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| `google/gemma-4-26B-A4B-it`         | NF4 4-bit | **77.7 %** | **68.0 %** |  39.5 %    | **57.9 %**  |  13.3 % ⚠︎ | **51.3 %** | **+26.7** |
| `google/gemma-4-E2B-it`             | NF4 4-bit |  42.8 %    |  26.2 %    | **41.7 %** |  44.1 %     |  51.6 % ⚠︎ |  41.1 %    | +16.4    |
| `Qwen/Qwen3.5-2B`                   | NF4 4-bit |  29.1 %    |  17.6 %    | —          |  47.0 %     |  51.1 % ⚠︎ |  33.6 %    |  +9.1    |
| `sarvamai/sarvam-1`                 | NF4 4-bit |  29.5 %    |   2.9 %    | —          |  32.9 %     |  49.2 % ⚠︎ |  28.6 %    |  +4.2    |
| `Qwen/Qwen3-0.6B`                   | NF4 4-bit |  19.6 %    |   2.7 %    |  23.8 %    |  19.1 %     |  29.4 %    |  18.9 %    |  −5.6    |

`—` = split skipped in that run (HellaSwag is 10,042 rows and was dropped from the shorter runs).
`⚠︎` = score affected by one of the known issues called out below — treat with care, not as the model's true ceiling on that task.

### Cross-model patterns to read with caution

1. **Winogrande's option1 / option2 labels look swapped in the translated dataset.** Three independent models (Gemma 2B IT, Qwen3.5 2B, Gemma 26B) all produce failures of the same shape: the model writes the *content* of option 2 (`"...**2: Maria**"`, `"...**2: Hunter**"`, `"...**2: house**"`) while the gold says 1. Three different decoders making the same systematic error on the same rows is much more likely a data issue than a coordinated model weakness. Suggested triage:

   ```python
   from odia_eval import show_failures
   show_failures(all_results["winogrande/validation"], n=20, max_chars=400)
   ```

   If the model's chosen name is the *correct* referent in the Odia sentence, fix the translation pipeline (probably the `option1` / `option2` columns are being written in upstream order while `answer_idx` is being re-indexed against the shuffled order).

2. **Gemma 4 26B's 13.3 % on Winogrande is a *different* failure than the label flip.** Most rows have `extracted=''` because the model emits a long Odia preamble ("ଏହି ବାକ୍ୟଟି ବ୍ୟାକରଣଗତ ଭାବେ ଅସ୍ପଷ୍ଟ ଅଟେ, କିନ୍ତୁ …") that overruns the 32-token MCQ budget before producing "1" or "2". Bumping `max_new_tokens` for verbose chat models (or enabling `reasoning=True` with a larger budget) would recover most of this. The MEAN of 51.3 % therefore *understates* this model's true OdiaBench score by several points.

3. **HellaSwag is capped by English distractors.** Three of the four endings are not translated; the model picks them by literally emitting the English text (`'D: stands on his hands and springs.'`). Translating all four endings is the V2 fix noted in the package README.

4. **Base (non-instruction-tuned) models surface as `extracted=''` or HTML tag salad.** Sarvam-1 emits `</stitle>` / `[INST]` markup; Qwen3-0.6B opens every answer with English `<think>\nOkay, let's…` and never closes. Neither is a fair comparison against IT models on a chat-style benchmark.

---

## `google/gemma-4-26B-A4B-it`

Gemma 4 26B-A4B (MoE), 4-bit NF4 via BitsAndBytes. Full splits, greedy decoding, no self-consistency.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  911 / 1172     | 77.7 %   | 75.3 – 80.0   | 25.1 % | **+52.6** |
| gsm8k / test             |  897 / 1319     | 68.0 %   | 65.4 – 70.5   |  0.0 % | **+68.0** |
| hellaswag / validation   | 3962 / 10042    | 39.5 %   | 38.5 – 40.4   | 25.0 % | **+14.5** |
| truthfulqa / validation  |  473 /  817     | 57.9 %   | 54.5 – 61.2   | 22.6 % | **+35.3** |
| winogrande / validation  |  168 / 1267     | 13.3 % ⚠︎ | 11.5 – 15.2  | 50.0 % |  −36.7    |
| **MEAN (unweighted)**    |                 | **51.3 %** |             |        | **+26.7** |

Headline: strongest model on the board for arc / gsm8k / truthfulqa. The winogrande number is a generation-length artefact (see caveat 2 above), not a real reasoning regression.

---

## `google/gemma-4-E2B-it`

Gemma 4 2B IT, 4-bit NF4 via BitsAndBytes. Full splits, greedy decoding, no self-consistency.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  502 / 1172     | 42.8 %   | 40.0 – 45.7   | 25.1 % | **+17.7** |
| arc / validation         |  120 /  299     | 40.1 %   | 34.7 – 45.8   | 25.1 % | **+15.0** |
| gsm8k / test             |  345 / 1319     | 26.2 %   | 23.9 – 28.6   |  0.0 % | **+26.2** |
| hellaswag / validation   | 4183 / 10042    | 41.7 %   | 40.7 – 42.6   | 25.0 % | **+16.7** |
| truthfulqa / validation  |  360 /  817     | 44.1 %   | 40.7 – 47.5   | 22.6 % | **+21.5** |
| winogrande / validation  |  654 / 1267     | 51.6 % ⚠︎ | 48.9 – 54.4  | 50.0 % |   +1.6    |
| **MEAN (unweighted)**    |                 | **41.1 %** |             |        | **+16.4** |

Notes:
* HellaSwag (41.7 %) is the strongest 2B-class score on the board — Gemma 4 has the best cross-lingual ending-matching of the small models tested.
* GSM8K 26.2 % is a respectable 4-bit number; `n_votes>1` with `reasoning=True` and a stochastic generate_fn typically adds 3–6 points.

---

## `Qwen/Qwen3.5-2B`

Qwen 3.5 2B, 4-bit NF4 via BitsAndBytes. HellaSwag skipped in this run.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  341 / 1172     | 29.1 %   | 26.6 – 31.8   | 25.1 % |  +4.0    |
| arc / validation         |   70 /  299     | 23.4 %   | 19.0 – 28.5   | 25.1 % |  −1.7    |
| gsm8k / test             |  232 / 1319     | 17.6 %   | 15.6 – 19.7   |  0.0 % | **+17.6**|
| truthfulqa / validation  |  384 /  817     | 47.0 %   | 43.6 – 50.4   | 22.6 % | **+24.4**|
| winogrande / validation  |  647 / 1267     | 51.1 % ⚠︎ | 48.3 – 53.8  | 50.0 % |  +1.1    |
| **MEAN (unweighted)**    |                 | **33.6 %** |             |        |  +9.1    |

TruthfulQA 47.0 % is notable for a 2B model — close to Gemma 4 2B IT's 44.1 % on the same split. ARC scores trail Gemma; GSM8K reasoning works but at half the rate.

---

## `sarvamai/sarvam-1`

Sarvam 1 (Indic-specialist pretrain), 4-bit NF4. HellaSwag skipped.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  346 / 1172     | 29.5 %   | 27.0 – 32.2   | 25.1 % |  +4.4    |
| gsm8k / test             |   38 / 1319     |  2.9 %   |  2.1 –  3.9   |  0.0 % |  +2.9    |
| truthfulqa / validation  |  269 /  817     | 32.9 %   | 29.8 – 36.2   | 22.6 % | **+10.3**|
| winogrande / validation  |  623 / 1267     | 49.2 % ⚠︎ | 46.4 – 51.9  | 50.0 % |  −0.8    |
| **MEAN (unweighted)**    |                 | **28.6 %** |             |        |  +4.2    |

Output samples show base-model behaviour — `</stitle>`, `</stong>`, `[INST]` tag salad and abrupt single-letter answers. To compare fairly against IT models, a chat-template wrapper or an instruction-tuned variant is needed.

---

## `Qwen/Qwen3-0.6B`

Qwen 3 0.6B base, 4-bit NF4. Full splits.

| Benchmark                | Correct / Total | Accuracy | 95% Wilson CI | Chance | Δ vs chance |
|---|--:|--:|--:|--:|--:|
| arc / test               |  230 / 1172     | 19.6 %   | 17.5 – 22.0   | 25.1 % |  −5.5    |
| gsm8k / test             |   36 / 1319     |  2.7 %   |  2.0 –  3.8   |  0.0 % |  +2.7    |
| hellaswag / validation   | 2392 / 10042    | 23.8 %   | 23.0 – 24.7   | 25.0 % |  −1.2    |
| truthfulqa / validation  |  156 /  817     | 19.1 %   | 16.5 – 21.9   | 22.6 % |  −3.5    |
| winogrande / validation  |  373 / 1267     | 29.4 %   | 27.0 – 32.0   | 50.0 % | **−20.6**|
| **MEAN (unweighted)**    |                 | **18.9 %** |             |        |  −5.6    |

Below chance on most splits. Every output opens with an English `<think>\nOkay, let's…` block and never recovers in Odia — the 0.6B parameter budget plus default thinking mode is below the floor needed for Odia reasoning at this prompt format. Listed for the lower bound.

---

## Run metadata (common)

| Key | Value |
|---|---|
| `odia_eval` version       | 0.4.0 |
| Quantisation              | NF4 4-bit (BitsAndBytes, double-quant, fp16 / bf16 compute depending on GPU) |
| `attn_implementation`     | auto → flash_attention_2 / sdpa fallback |
| `torch.compile`           | enabled, `mode="default"` (Colab-safe default in 0.4.0+) |
| `sort_by_length`          | False |
| `reasoning`               | False (no `<think>` / `\boxed{}` wrapping) |
| `n_votes`                 | 1 (greedy, no self-consistency) |
| Seed                      | 42 |
| Data source               | HuggingFace Hub `tripathysagar/odia-*` (parquet shards via `hub_run_eval`) |

### Known runtime caveats during these runs

* Dynamo occasionally hit `config.recompile_limit (8)` partway through `arc/test`, falling back to eager for the remainder of that split. Bump `torch._dynamo.config.recompile_limit = 64` in the compile cell to avoid this on subsequent runs.
* `cudagraph partition due to non gpu ops` spam from Inductor is silenced in 0.4.0 by raising the `torch._inductor.cudagraph_trees` logger to `ERROR`.
* `MODEL_ID = "Qwen/Qwen3.5-2B"` is not a published Qwen identifier as of this writing; double-check the model actually loaded against the intended weights before drawing comparative conclusions from that row.
