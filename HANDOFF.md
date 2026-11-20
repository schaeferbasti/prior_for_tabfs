# Handoff: TabFS with Kumo encoder-only importance (branch `test_2`)

*Last updated 2026-11-20 · branch `test_2`*

## TL;DR

- **Goal:** one importance score per feature for a table `(X, y)`, from a pretrained tabular foundation model.
- **Setup:** a port of the co-worker's working method (private repo `m-herre/kumo-cls-studies`, studies of 2026-11-13).
  - **Model:** NVIDIA **Kumo-Tabular Large**, pretrained, **frozen**. Nothing is trained.
  - **Importance of feature j:** shuffle column j across the context rows, run only Kumo's encoder (the 4 readout/CLS
    tokens per row), and measure `mean ||E_shuffled − E|| / mean ||E||`.
  - **Selection on real tables:** keep a feature if its importance beats a null: a shuffled copy of itself
    (`rule="shadow"`, default) or the stronger of two Gaussian noise columns (`rule="noise"`).
- **The previous TabICLv2 + trained CLS-attention readout was removed** (it didn't learn; see git history at
  `2d231d0`). The v1 nanoTabPFN handoff is the untracked `HANDOFF_test.md`.

## Why this replaced the TabICL-CLS version

| | Old (TabICLv2 CLS) | Now (co-worker's Kumo method) |
|---|---|---|
| Signal | pre-softmax attention logits of CLS #0 over feature tokens | the CLS **outputs** (full row embedding) |
| Attribution | single pass, grouped tokens averaged (3 features per token) | counterfactual: shuffle one column, d+1 passes |
| Training | fine-tuned 9-parameter readout (didn't learn) | none (zero-shot) |
| Zero-shot AUC on prior | 0.60 (CLS #0) | 0.84 (co-worker, vs. 0.81 for \|Spearman\|) |

## Co-worker's reference results (Kumo-Large, CUDA)

- **Prior** (4380 tasks, 1414 with both relevant and independent features): per-task AUC **0.839**, baseline
  |Spearman(x, y)| **0.810**. 87% of features are relevant.
- **Real data, 89 BeyondArena tasks:** Spearman with CatBoost SHAP 0.31 (mutual information 0.55). Removal-curve
  mean rank: SHAP 2.03, permutation 2.21, MI 3.02, **Kumo 3.59**, random 4.15. The signal is label-driven:
  it collapses with shuffled labels (0.09) and with random weights (0.11).
- **Downstream, CatBoost on 108 tasks:** shadow rule vs. all features: geomean error ratio 1.0007 (tie); noise rule 1.016.
- So: a sound, training-free baseline, but **not better than simple statistics** on real data.

## Repo map

| Path | What it is |
|---|---|
| `utils/fs/kumo_fs.py` | Load Kumo (`sdm`), `encoder_importance` (tensors), `KumoEncoder` + `select_features` (pandas, Kumo's recipe) |
| `utils/prior/prior.py` | Graph-SCM prior tasks with "shares an ancestor with y" labels (`sample_dataset`, `iter_tasks`) |
| `evaluation/run_evaluation_pipeline.py` | Per-task AUC of Kumo importance vs. \|Spearman\| on prior tasks, plus the summary |
| `prediction/run_prediction_pipeline.py` | Selection on the 3 test datasets → `data/test_results/<name>_results.parquet` |

Source mapping to the co-worker's repo:
- `encoder_importance`, `sample_dataset`, the evaluation script ← `2026-11-13-encoder-importance-prior/{run,analyse}.py`
- `KumoEncoder`, `rule="noise"` ← `2026-11-13-encoder-select-catboost/select_features.py`
- `rule="shadow"` ← `2026-11-13-encoder-select-catboost/select_v2.py`

## Setup

- Conda env `feature-selection-benchmark`, plus NVIDIA's `sdm`, pinned like TabArena:
  `pip install --no-deps "git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19"`
  (installed 2026-11-20; its dependencies torch/numpy/pyarrow/huggingface_hub/safetensors were already there).
- Weights: `nvidia/Kumo-Tabular` on Hugging Face (`large/classifier.pt`, `large/regressor.pt`, 855 MB each), Hub tag
  `v1.0.1`, downloaded on first use. If the download hangs at 0 bytes, set `HF_HUB_DISABLE_XET=1`.
- `tabicl==2.2.0` provides the graph-SCM prior (`tabicl.prior.graph_lib`).

```bash
python evaluation/run_evaluation_pipeline.py 0 100   # worker 0, 100 tasks -> data/eval_results/worker0.jsonl + summary
python prediction/run_prediction_pipeline.py         # 3 test datasets
```

## Differences from the co-worker's setup (could not be cloned exactly)

1. **`STAGE_CONFIG`:** theirs is in the private `kumo_prune` package. Here: `PriorConfig` defaults with
   `filter_unpredictable_graphs`, `filter_unpredictable_datasets` and `remove_trivial_datasets` on.
   **Ask the co-worker for the exact config.** With this one, 79% of features are relevant and 40% of tasks are
   scorable (20-task check), against 87% / 32% for theirs.
2. **Prior code:** `tabicl.prior.graph_lib` (tabicl 2.2.0) instead of their vendored copy
   `kumo_prune/_vendor/nanotabiclv2/prior.py`. Same classes and call sequence, but the versions may differ.
3. **Device:** CPU / fp32 here, CUDA / fp16 autocast for them. Expect small numeric differences and much slower runs.
4. **Real-data input:** their selector ran on TabArena/BeyondArena tasks (AutoGluon pipeline, official split 0);
   here it runs on the 3 local test parquet files with the same column typing rules.
5. Their other 11 studies (geometry probes, feature decoders, CatBoost runs, timing) are not ported.

## Next steps

1. Get `STAGE_CONFIG` from the co-worker and re-run the evaluation; compare with AUC 0.839 / 0.810.
2. Use this as the zero-shot baseline. Ideas to go beyond it: per-token importance (the 4 readout tokens
   separately), importance on the ICL output instead of the encoder, or a trained head on the shuffle deltas.
