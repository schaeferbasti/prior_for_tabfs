# Prior Test for Feature Selection with a Tabular Foundation Model

Feature importance from a pretrained tabular foundation model, without refitting a predictor per feature.

## Concept (branch `test_2`)
A port of the co-worker's Kumo CLS-token studies (`m-herre/kumo-cls-studies`, 2026-11-13):
- **Model:** NVIDIA **Kumo-Tabular Large**, pretrained and frozen. Each row is summarized by 4 readout (CLS) tokens.
- **Importance of feature j:** shuffle column j across the context rows and measure the relative change of the
  row embeddings (encoder only, no ICL stage). See `utils/fs/kumo_fs.py`.
- **Selection:** keep a feature if it beats a shuffled copy of itself (`rule="shadow"`) or two noise columns (`rule="noise"`).
- **Evaluation on synthetic data:** graph-SCM prior tasks with known relevance (feature shares an ancestor with y),
  per-task AUC against |Spearman(x, y)|.

Earlier versions: nanoTabPFN + LOCO (commit `3b1523e`, see `HANDOFF_test.md`), TabICLv2 + trained CLS-attention
readout (commit `2d231d0`). Details and caveats: `HANDOFF.md`.

## Setup
Conda env with `torch`, `tabicl==2.2.0`, `scikit-learn`, `pandas`, `pyarrow`, `scipy`, plus NVIDIA's `sdm`:

    pip install --no-deps "git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19"

Kumo weights (`nvidia/Kumo-Tabular`, ~855 MB per task) are downloaded from Hugging Face on first use.

## Run
    python evaluation/run_evaluation_pipeline.py [worker_id] [n_tasks]   # prior tasks -> data/eval_results/
    python prediction/run_prediction_pipeline.py                         # test datasets -> data/test_results/

## To Do
- [X] Kumo encoder-only shuffle importance (prior and real tables)
- [X] Prior tasks with ground-truth relevance; AUC vs. |Spearman| baseline
- [ ] Exact `STAGE_CONFIG` from the co-worker
- [ ] Downstream evaluation (accuracy after selection) vs. MI, SHAP, permutation importance
