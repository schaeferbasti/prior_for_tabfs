# Prior Test for Feature Selection with a Tabular Foundation Model

This project studies a prior for feature selection with tabular foundation models: the model is
pretrained so that **one forward pass over a table `(X, y)` returns one importance score per feature**.
The supervision is leave-one-column-out (LOCO) importance, computed on every synthetic prior task.

## Concept (branch `test_2`, v2)
- **Backbone:** pretrained **TabICLv2** (`tabicl` package), the parent architecture of NVIDIA's Kumo Tabular.
  Each row is summarized by **4 CLS tokens** in the row-wise transformer.
- **Readout:** the importance of feature *f* is the pre-softmax attention logit of **CLS token #0** over
  feature *f* in the last row block, averaged over rows and combined over heads with learned weights
  (see `utils/tfm/tabicl_fs.py`).
- **Prior:** synthetic SCM tasks from TabICL's prior (Kumo's generator is not released). Each feature gets a
  **ground-truth relevance label from the generating causal graph** (not d-separated from y given the other
  features, `utils/prior/relevance.py`). LOCO targets from a Random-Forest teacher are optional (`utils/fs/fs.py`).
- **Training objectives** (`OBJECTIVE` in the pretraining script):
  `relevance` = BCE against the generator labels; `gating` = no labels, the CLS scores gate the features
  (hard-concrete, gated-out features become noise) and TabICL must still predict y, plus an L0 penalty;
  `loco` = regression on LOCO. Supervised objectives add λ · TabICL's own cross-entropy.

The original nanoTabPFN version (row-collapse head) is at commit `3b1523e`; `concept/TabFS.png` shows that version.

## Setup
Conda env with `torch`, `tabicl>=2.2`, `scikit-learn`, `h5py`, `pandas`, `pyarrow`, `scipy`, `joblib`.
The TabICLv2 checkpoint (`tabicl-classifier-v2-20260212.ckpt`, ~105 MB) is downloaded from Hugging Face on first use.

## Pretraining
`python pretraining/run_pretraining_pipeline.py`

Builds `data/prior/fs_prior_tabicl.h5` if missing (~2.5 min for 4000 tasks without LOCO), then fine-tunes
and saves `utils/tfm/tabicl_fs_<objective>_<dd_mm_yyyy>.pt`. Settings are constants at the top of the script.

## Inference
`python prediction/run_prediction_pipeline.py`

Writes a ranked feature table per test dataset to `data/test_results/<name>_results.parquet`.
`CHECKPOINT = None` runs the zero-shot baseline (pretrained TabICLv2, untrained readout).

## To Do
- [X] Prior from the TabICL SCM prior with ground-truth relevance labels (LOCO optional)
- [X] TabICLv2 backbone with CLS-token importance readout
- [X] Objectives: relevance (supervised), gating (self-supervised), LOCO; held-out AUC vs. relevance
- [ ] Full fine-tuning runs per objective; tune `LAMBDA_PRED`, `LAMBDA_L0`, learning rates, which CLS token
- [ ] Prior with more irrelevant features (currently ~85% of features are relevant)
- [ ] Evaluation pipeline (`evaluation/`): downstream accuracy after top-k selection vs. baselines
- [ ] Optional: port the readout to Kumo Tabular once its training code is released
