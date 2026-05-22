# OdiaBench Notebooks

Runnable Jupyter notebooks that exercise the [`odia_eval`](https://github.com/tripathysagar/odia_eval) library.

| Notebook | Purpose | Runtime | One-click |
|---|---|---|---|
| [`capability.ipynb`](./capability.ipynb) | Tour the `odia_eval` API — load benchmarks, build prompts, score, report (Wilson CIs + chance baselines), save/load, inspect failures | **CPU, ~30 s** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/capability.ipynb) |
| [`eval_odiabench.ipynb`](./eval_odiabench.ipynb) | Full 4-bit Gemma reference evaluation across all 5 benchmarks (local jsonl via cloned OdiaBench) | **GPU**, ~5–30 min depending on hardware | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/eval_odiabench.ipynb) |
| [`eval_odiabench_working.ipynb`](./eval_odiabench_working.ipynb) | Same eval but data is streamed from the `tripathysagar/odia-*` HF Hub repos, with multi-split support and `n_votes` self-consistency | **GPU**, ~5–30 min | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripathysagar/odia_eval/blob/main/notebooks/eval_odiabench_working.ipynb) |

Both notebooks auto-detect their runtime (local / Colab / Kaggle).
They `pip install odia_eval` from its [standalone repo](https://github.com/tripathysagar/odia_eval) and `git clone` OdiaBench for the benchmark data files (`dataset/data/*.jsonl`).
No manual path setup required.

## Where can these notebooks run?

| Platform | Status | How |
|---|---|---|
| **Local Jupyter / VS Code** | ✓ | `jupyter notebook notebooks/` from the repo root |
| **Google Colab** | ✓ | Click the Colab badge above. The first cell `pip install`s `odia_eval` and clones the data. `eval_odiabench` needs a GPU runtime (Runtime → Change runtime type → T4 / A100). |
| **Kaggle Kernels** | ✓ | Create a new Notebook, set Internet **on**, paste the GitHub URL into the cell from the README quick-start, or upload the `.ipynb` directly. |
| **HuggingFace Spaces** | ✗ | Spaces host interactive apps (Gradio / Streamlit / Docker). They don't execute notebooks. To deploy `odia_eval` on a Space, wrap `run_eval` in a Gradio app and add `odia_eval @ git+https://github.com/tripathysagar/odia_eval.git` to the Space's `requirements.txt`. |
| **HuggingFace dataset / model repo** | view-only | A notebook checked into a HF repo is rendered via [nbviewer](https://nbviewer.org/) — readers can read it but not run it. |
| **HuggingFace Inference Endpoints** | indirect | You can call an Endpoint from `generate_fn` inside either notebook; the notebook itself still runs on Colab / Kaggle / local. |

## Local quick start

```bash
git clone https://github.com/tripathysagar/OdiaBench.git
cd OdiaBench

# Install odia_eval (zero required deps; optional extras: hub, pandas, model, all)
pip install "odia_eval @ git+https://github.com/tripathysagar/odia_eval.git"

jupyter notebook notebooks/capability.ipynb
```

The capability notebook is pure stdlib — no extras needed. For the model-eval notebook install the `[model]` or `[all]` extra:

```bash
pip install "odia_eval[all] @ git+https://github.com/tripathysagar/odia_eval.git"
```

## Colab tips

* `eval_odiabench` — switch to a GPU runtime **before** running cell 2 (the install cell). Cold install on a T4 takes ~3 min.
* `capability` — runs fine on a CPU runtime; finishes in under a minute.
* Both notebooks `pip install odia_eval` and `git clone` OdiaBench into `/content/OdiaBench` on first run. Re-running the prelude cell is a no-op.

## Kaggle tips

* Enable **Internet** under Settings → Notebook options (required for `git clone` and HF Hub).
* For `eval_odiabench`, pick a GPU accelerator (T4 ×2 is the most common free tier).
* Both notebooks place the repo at `/kaggle/working/OdiaBench`, so anything you save under `dataset/data/` or `results/` persists for the session.
