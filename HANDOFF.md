# Handoff: TabFS, feature selection in one forward pass

*Last updated 2026-11-19 · branch `test_2` · v2 (TabICLv2 + CLS readout), uncommitted on top of `3b1523e`*

## TL;DR

- **Goal:** a tabular foundation model that returns **one feature-selection score per feature** from a single forward pass over `(X, y)`.
- **Model (v2):**
  - **Backbone:** pretrained **TabICLv2**, the parent architecture of NVIDIA's *Kumo Tabular*. Kumo's training code and prior are not released.
  - **Readout:** the score of feature *f* is how strongly **CLS token #0** attends to *f* in the last row block, taken before the softmax and averaged over rows.
- **Targets:** LOCO is no longer the default target, because it was mostly noise (see [History](#history)). There are two replacements:
  1. **`relevance` (default):** supervised training on **ground-truth relevance labels from the data generator**. A feature is relevant iff it is not d-separated from y given the other features in the causal graph that generated the task. The labels are noise-free and cost nothing: a 1000-task prior takes 34 s, against ~8 min with LOCO.
  2. **`gating`:** **self-supervised**, no feature labels. The CLS scores open or close a gate per feature. Closed features are replaced by noise, TabICL must still predict y, and an L0 penalty pushes towards keeping few features. This tests the idea that "the CLS token is enough".
  - **`loco`** is still available for comparison.
- **Status:**
  - Everything is verified: generator labels, d-separation, and the readout; overfit checks pass.
  - **Short training runs did not learn** (per-task AUC stays at ~0.58–0.60). The [diagnosis](#diagnosis-why-nothing-is-learned-2026-11-19) shows two causes:
    1. **The signal is too permissive.** 85% of features are "relevant", many with negligible effect, so even a direct ExtraTrees importance only reaches 0.72 per task.
    2. **The loss lets the model learn task-level offsets instead of within-task rankings.** Per-task AUC doesn't improve even on *training* tasks.
  - **Update (F-test check):** the setup cannot even learn the ANOVA F-statistic, a deterministic function of the visible data. Spearman plateaus at ~0.42 where ~1.0 is possible ([details](#f-test-learnability-check-objective--ftest)). So **learning / the readout is the first problem**, and the signal question stays open until that check passes.
- **Next:** run the flexible-head F-test check to see whether the CLS-attention readout is the bottleneck. Then fix the readout, then the loss and the prior (see [next steps](#suggested-next-steps)).

---

## The model

```
          ┌───────────── column-wise embedding (induced attention over rows, target-aware) ─────────────┐
X, y_train│                                                                                            │
          └──> row-wise transformer: [CLS0 CLS1 CLS2 CLS3 | tok_1 … tok_H]  (3 blocks, RoPE)            │
                     │                                                                                  │
                     │ last block: CLS queries attend to all tokens                                      │
                     ├──> CLS0 · k_tok / √d  (pre-softmax logits) ──> tokens→features ──> mean over rows │
                     │                                         ──> Σ_h w_h · (…) + b  = score (H,)      │
                     └──> concat(CLS0..3) = row embedding ──> ICL transformer ──> class logits (test rows)
```

| Stage | Parameters | Used for the score? |
|---|---|---|
| Column embedding (`col_embedder`) | 0.9 M | yes |
| Row transformer with 4 CLS (`row_interactor`) | 0.4 M | yes (CLS #0's attention logits) |
| ICL transformer (`icl_predictor`) | 26.3 M (95%) | **no**. Only in training: the prediction loss for `relevance`/`loco` (weight `LAMBDA_PRED`), and the main loss for `gating` |
| Readout (per-head weights + bias) | 9 | yes |

- **Inference** (`feature_importance`) runs only the first two stages. All rows are labelled context, and the ICL transformer is skipped.
- **What the readout is.** The score is the CLS token's *attention over features*, not the CLS embedding.
- **Feature groups.** TabICLv2 does not give each feature its own token: **token g holds features g+1, g+2 and g+4 (mod H)**. Token logits are therefore mapped back to features by averaging over the 3 tokens that contain each feature.
- **Interpreting the score** (`scores_to_selection`):

  | Objective | Score means | Selected if |
  |---|---|---|
  | `relevance` | logit of P(relevant) | P > 0.5 |
  | `gating` | gate log-α | deterministic gate > 0 |
  | `loco` | LOCO units | > 0 |

---

## Ground-truth relevance labels: `utils/prior/relevance.py`

**Definition.** A feature X_j is **relevant ⇔ X_j is not d-separated from y given all other observed features**. This is *strong relevance* (Kohavi & John). Under faithfulness it is exactly the **Markov boundary** of y, the set an optimal feature selector has to keep.

**How it works**
- `RelevanceSCMPrior` subclasses TabICL's `SCMPrior` and overrides `generate_dataset`. Wherever TabICL code had to be copied (`handle_outputs`, `Reg2Cls._num2cat`/`_process_features`), the copies make the **same random draws**.
  - **Verified:** with the same seed it produces **bit-identical tasks** to `tabicl`'s own prior. So the task distribution is unchanged; it only adds labels.
- **Graph construction:**
  - **MLP SCM:** nodes are the root causes plus every hidden unit; an edge i→j exists iff weight W[j,i] ≠ 0 (dropout zeroes weights). All activations in `tabicl.prior` act element-wise; I checked every class.
  - **Tree SCM:** an edge i→j exists iff input i is used in a split of the tree model producing output j. XGBoost's multi-output trees share splits across all outputs.
  - **X and y** are the hidden units chosen in `handle_outputs` (recorded).
  - **Categorical conversion** makes the observed variable a new child of the unit: conditioning on a coarsened feature doesn't block paths.
  - **Columns** follow Reg2Cls' random permutation and the deletion of constant columns.
- **D-separation** uses the moral graph of the ancestral set of all observed variables. That set is the same for every feature test, so one moralization per task is enough.
  - **Verified** on textbook cases: chain, spouse, latent confounder, isolated feature, latent mediator.
  - Cost: ~34 s for 1000 tasks, including generation (tree SCMs dominate).
- **Sanity check against usefulness** (64 tasks, LOCO with 5 repeats):
  - Relevant features have mean LOCO **+0.028**, irrelevant ones **−0.044**.
  - LOCO separates the two groups with AUC 0.70 pooled (0.63 per task).
  - So the labels match actual predictive usefulness, though not perfectly. Part of the gap is LOCO noise, part is the caveats below.

**Caveats**
- **The labels are structural.** A weak edge, a saturated activation or a near-deterministic unit can make a "relevant" feature useless in a finite sample (faithfulness violations).
- **The labels are skewed.** In this prior **85% of features are relevant**, and **58% of tasks have every feature relevant**: a dense SCM connects almost everything. Training therefore uses class-balanced BCE (`pos_weight`), and evaluation uses AUC.
  - Next step: make the prior more selection-like, e.g. sparser SCMs or added distractor columns labelled irrelevant.

---

## Repo map

| Path | What it is |
|---|---|
| [`utils/tfm/tabicl_fs.py`](utils/tfm/tabicl_fs.py) | `TabICLFS`: TabICLv2, the CLS-attention readout, `forward_gated` (hard-concrete gates), `scores_to_selection`, checkpoint I/O |
| [`utils/tfm/train.py`](utils/tfm/train.py) | Fine-tuning for the 3 objectives; `evaluate` (AUC vs. relevance for every objective) |
| [`utils/prior/relevance.py`](utils/prior/relevance.py) | `RelevanceSCMPrior` (records the graph; same tasks as `tabicl`) and `strongly_relevant` (d-separation) |
| [`utils/prior/prior.py`](utils/prior/prior.py) | `build_prior` (tasks + relevance, optional LOCO → HDF5) and `PriorTasks` (loader, split, iterate) |
| [`utils/fs/fs.py`](utils/fs/fs.py) | LOCO teacher (RF, averaged over `n_repeats` splits); only for `OBJECTIVE="loco"` and for checks |
| [`pretraining/run_pretraining_pipeline.py`](pretraining/run_pretraining_pipeline.py) | Builds the prior if needed, then fine-tunes. **All settings, including `OBJECTIVE`, are constants at the top** |
| [`prediction/run_prediction_pipeline.py`](prediction/run_prediction_pipeline.py) | One forward pass per test dataset: score, probability or gate value, and selected/ranked features |
| `evaluation/run_evaluation_pipeline.py` | Still **empty**, untracked |
| `data/prior/fs_prior_tabicl.h5` | The v2 prior; created by the pretraining script, untracked |
| `data/prior/300k_150x5_2.h5`, `loco_prior.h5`, `new_prior.h5`, `utils/tfm/weights_*.pt`, `concept/TabFS.png` | **v1 / `MetaTabFS` leftovers, not used** |

---

## Setup and running

- **Environment:** the conda env `feature-selection-benchmark`. **Pin `tabicl==2.2.0`**: the code uses and partly mirrors TabICL internals (`_train_forward`, `_icl_predictions`, `SCMPrior.generate_dataset`, `Reg2Cls`, `tabicl._sklearn.preprocessing`). After a `tabicl` upgrade, re-run the bit-identical-prior check (see [Verification](#verification-done)).
- **Pretrained checkpoint:** `tabicl-classifier-v2-20260212.ckpt` (27.5M params) downloads automatically from Hugging Face and is cached on this machine.
- **Running the scripts:** they add the repo root to `sys.path`, so they run from anywhere.

```bash
python pretraining/run_pretraining_pipeline.py   # OBJECTIVE = "relevance" | "gating" | "loco" at the top
python prediction/run_prediction_pipeline.py     # all 3 test datasets -> data/test_results/<name>_results.parquet
```

**Timings** (CPU, 12 cores):

| Step | Time |
|---|---|
| Prior with 4000 tasks | ~2.5 min (with LOCO: ~30 min) |
| Fine-tuning | ~0.25 s per task for `relevance`/`loco`, ~0.5 s per task for `gating` (two passes). 500 steps × 8 tasks ≈ 17 / 35 min |
| Prediction | ~8 s for all 3 datasets, ~2 GB peak (1991-feature dataset) |

**Prediction checkpoint options** (`CHECKPOINT`):
- `"latest"`: the newest `utils/tfm/tabicl_fs_*.pt`.
- A path: that checkpoint.
- `None`: the **zero-shot** pretrained TabICLv2, with its score read as a relevance logit.

---

## Training objectives: `utils/tfm/train.py`

| Objective | Loss per task | Needs |
|---|---|---|
| `relevance` | `BCE(score, relevance, pos_weight=(1-p)/p) + λ_pred · CE(TabICL)` | generator labels |
| `gating` | `CE(TabICL on gated X) + λ_L0 · mean P(gate open)` | nothing |
| `loco` | `SmoothL1(score, LOCO / std) + λ_pred · CE(TabICL)` | LOCO prior (slow) |
| `ftest` | `MSE(score, per-task z-score of log1p(ANOVA F on the context rows)) + λ_pred · CE` | nothing. **A learnability check, not a selection target** |

- **Gating** (`forward_gated`):
  1. Pass 1 computes the gate log-α from the CLS scores.
  2. A hard-concrete gate z ∈ [0,1] is sampled per feature (Louizos et al., 2018).
  3. Pass 2 sees `z·x + (1−z)·ε` with ε ~ N(0,1) and predicts the test rows.
  - Noise replacement is necessary: TabICL's column embedding is scale-invariant, so a purely multiplicative gate would not remove any information.
  - The optimum keeps a minimal sufficient feature set, which is again the Markov boundary. That makes the generator labels a fair *evaluation* for gating, even though gating never sees them.
- **Shared settings:** AdamW, lr 1e-5 for the backbone and 1e-2 for the readout, gradient clipping 1, 8 tasks per step, `λ_pred = 0.1`, `λ_L0 = 0.1`. `FREEZE_BACKBONE=True` trains only the 9 readout parameters.
- **Validation** runs every `eval_every` steps and at step 0, where step 0 is the untrained model. It reports:
  - for every objective: **AUC vs. relevance** (pooled and per task) and the selected fraction
  - for gating: the gated CE
  - for LOCO: Spearman and MAE against a constant baseline

---

## Results so far

All runs below use a 1000-task prior (900 train / 100 val), CPU, default settings unless stated.

**Overfit checks** (4 tasks, backbone lr 1e-4):
- `relevance`: AUC **0.80 → 1.00** in 30 steps.
- `loco`: Spearman **0.09 → 0.98**.

**Zero-shot: is the pretrained CLS attention already a feature selector?**
- Setup: AUC of the *untrained* readout (equal head weights) against the generator labels, on 100 validation tasks. Per-task AUC uses the 42% of tasks that have both labels.
- Answer: **only weakly**. CLS #0/#1 carry some signal and CLS #3 is inverted: it attends to *irrelevant* features.

| CLS token | AUC pooled | AUC per task |
|---|---|---|
| #0 (default) | 0.61 | 0.60 |
| #1 | 0.61 | 0.61 |
| #2 | 0.54 | 0.47 |
| #3 | 0.41 | 0.39 |

**Short training runs** (150 steps × 8 tasks, backbone lr 1e-5; stopped early because nothing was learned):

| Run | step 0 (pooled / per task) | step 50 | step 100 | step 150 |
|---|---|---|---|---|
| `relevance` | 0.611 / 0.596 | 0.609 / 0.582 | 0.616 / 0.568 | 0.632 / 0.580 |
| `gating` (λ_L0 = 0.1) | 0.611 / 0.596 | 0.628 / 0.607 | stopped | stopped |

For gating at step 50, 71% of features were kept and the gated CE stayed flat (0.844 → 0.839).

### Diagnosis: why nothing is learned (2026-11-19)

All numbers below come from the same split (900 train / 100 validation tasks). "Per task" means AUC computed within each task, which is what feature selection needs. It is averaged over the **only 40 validation tasks that have both labels**, so its standard error is about ±0.03.

| Test | Pooled AUC | Per-task AUC |
|---|---|---|
| **D1: signal from data alone (no TFM)** | | |
| mutual information | 0.55 | 0.54 |
| ANOVA F | 0.58 | 0.56 |
| ExtraTrees impurity importance (MDI) | 0.66 | **0.72** |
| meta-model (GBM on MI/F/MDI + ranks, trained on 900 tasks) | 0.75 | 0.62 |
| **D2: does fine-tuning learn?** | | |
| zero-shot, *train* tasks | 0.63 | 0.59 |
| after 150 steps, *train* tasks | 0.69 | **0.58** |
| after 150 steps, val tasks | 0.63 | 0.58 |
| **D3: frozen-backbone probes (logistic regression; GBM is not better)** | | |
| CLS #0 per-head logits (what the readout sees) | 0.62 | 0.57 |
| all 4 CLS tokens' logits | 0.60 | 0.56 |
| feature-token embeddings (128-d) | 0.66 | 0.57 |
| logits + embeddings | 0.68 | 0.60 |

**Conclusions**

1. **The signal is weak and too permissive (main problem).**
   - Even a direct data method only reaches ~0.72 per task.
   - "Structurally relevant" includes every feature with *any* path to y. In these dense SCMs that is 85% of features, many with negligible effect, so the label mostly means "connected somewhere", not "matters".
2. **The model is not learning *within-task* ranking.**
   - After fine-tuning, per-task AUC is unchanged even on the *training* tasks (0.59 → 0.58), so this is underfitting, not overfitting.
   - The pooled gain on train (0.63 → 0.69) is a **task-level offset**: 58% of tasks are all-relevant, and the BCE rewards scoring whole tasks up or down rather than ranking features. That is a loss-design problem.
   - The backbone learning rate (1e-5) is probably also too low. The overfit check only worked at 1e-4.
3. **The frozen representation has little relevance information beyond simple statistics** (best probe 0.60 per task, below ExtraTrees MDI at 0.72). So a better readout alone won't fix it; the backbone has to learn.
4. **Data quantity is not the first bottleneck**, since the model doesn't even fit the training tasks. Data is cheap now anyway: 1000 tasks in 34 s.
5. **Evaluation was too small.** Per-task AUC over 40 tasks is noisy; differences below ~0.05 mean nothing.

### F-test learnability check (`OBJECTIVE = "ftest"`)

**Question:** can the setup learn *any* per-feature target? The target is the per-task z-scored log ANOVA F, computed on exactly the context rows the model sees. It is a deterministic function of the input, so a working setup should reach a Spearman correlation close to 1.

| Backbone lr | Spearman(score, F) on 100 val tasks: step 0 → 50 → 100 → 150 | Train MSE at step 150 (1.0 = predicting the mean) |
|---|---|---|
| 1e-5 | 0.26 → 0.29 → 0.35 → 0.36 | 0.86 |
| 1e-4 | 0.26 → 0.39 → 0.44 → 0.42 | 0.66 |

**Result: the setup cannot even fit a trivial, fully visible target.** A higher learning rate helps a little, but it plateaus at ~0.42. The failure on the relevance labels is therefore at least as much a **learning / readout problem** as a signal problem, and the signal question can't be answered until this check passes.

**Prime suspect: the readout.**
- It has 9 parameters.
- The score must come out of CLS #0's attention logits *through TabICLv2's feature groups*: every token mixes 3 features, and a feature's score averages 3 tokens.
- That makes it a circulant blur over the features, which the backbone has to learn to undo, and it is badly conditioned for some numbers of features.

**Next check:** fit the same F-test target with a flexible head, for example an MLP on per-feature embeddings, with the backbone frozen and unfrozen. If that reaches ~0.9 while the CLS readout stays at ~0.4, the readout is the bottleneck.

---

## Known issues and caveats

1. **The relevance labels are skewed (85% relevant).** See above. AUC is the metric that counts; "selected" fractions near 0.85 are expected.
2. **Context differs between training and inference.** During training the model sees labels for the prior's train split, 10–90% of rows. At inference all rows are labelled.
3. **Grouped tokens blur attribution.** A feature's score averages 3 tokens that each also contain 2 other features. For H < 5 the groups contain duplicates.
4. **Classifier only.** Regression targets are binned into ≤10 quantile classes at prediction. More than 10 classes go through TabICL's mixed-radix column embedding.
5. **Tasks are processed one at a time** (no padding, exact but slow). Grouping tasks by `(H, T, train_size)` would allow real batches on a GPU.
6. **Kumo itself is not used.** Its row stage has the same 4-CLS readout design, so the readout should port directly once fine-tuning Kumo is possible.
7. **Gating may need tuning:** `λ_L0`, and possibly temperature or annealing. It can collapse to all-open or all-closed gates. Watch the `selected` fraction and the gated CE together.

---

## Suggested next steps

These follow from the [diagnosis](#diagnosis-why-nothing-is-learned-2026-11-19) and the [F-test check](#f-test-learnability-check-objective--ftest). Do them in order and re-run the checks after each step.

0. **First, make the setup learn.**
   - Run the flexible-head F-test check: an MLP on per-feature embeddings, with the backbone frozen and unfrozen.
   - If the flexible head reaches ~0.9 Spearman, redesign the readout. Options: avoid the grouped-token blur, use the CLS output with a per-feature head, or use a backbone without feature grouping.
   - Do not continue until `OBJECTIVE="ftest"` reaches ~0.9 on validation.

1. **Make the positives detectable: label by effect size, not by any connection.**
   - The current label counts every feature with *any* path to y as relevant (85%). Many have negligible effect and cannot be found in 100–500 rows, which caps even ExtraTrees MDI at 0.72.
   - Fix: the generator can draw e.g. 20k extra rows from the **same SCM**. Compute a near noise-free importance on that large sample (drop-column or permutation; probably on the continuous y before `Reg2Cls`, since class thresholds are sample-dependent).
   - Then either label relevant = structurally relevant **and** importance above a threshold, or train on the graded importance with a ranking loss. This is LOCO done properly: large-sample and from the true generator.
2. **Fix the loss so that only within-task ranking pays off.**
   - Use a pairwise or listwise ranking loss within each task, and skip tasks without negatives (generate more tasks instead; they are cheap).
   - **Distractor features from an independent SCM are *not* recommended as the fix.** They would balance the labels, but:
     - they don't touch the too-permissive positives;
     - they only teach "unrelated to y vs. connected", which mutual information already does;
     - they add a shortcut: they correlate with each other but not with the real features, so they can be spotted without looking at y.
     - They are at most useful as an evaluation / robustness check.
3. **Fix the optimization.** Use backbone lr ~1e-4 (the overfit check only worked there), more steps, and more tasks (data is cheap). Before a long run, check that per-task AUC improves on *training* tasks.
4. **Fix the evaluation.**
   - Use ≥500 validation tasks.
   - Make per-task AUC the primary metric.
   - Report ExtraTrees MDI (0.72 per task) and zero-shot as reference lines in every validation log.
5. Only then: full `relevance` and `gating` runs, and sweeps over `IMPORTANCE_CLS`, `LAMBDA_PRED` and `LAMBDA_L0`.
6. Write `evaluation/run_evaluation_pipeline.py`:
   - Downstream accuracy after keeping the selected or top-k features.
   - Baselines: RF-LOCO, permutation importance, MDI, and zero-shot CLS.

The diagnostic script is not in the repo yet. Its logic is described in the diagnosis section; it is a candidate for `evaluation/`.

---

## Verification done

| Check | Result |
|---|---|
| Readout vs. backbone | Predictions and row representations bit-identical to unmodified TabICL; `softmax(logits) @ v` reproduces the real attention output to within 1e-6 |
| Token → feature mapping | Matches TabICL's `feature_grouping` |
| `RelevanceSCMPrior` | Bit-identical tasks to `tabicl.prior.SCMPrior` with the same seed |
| `strongly_relevant` | Correct on chain, spouse, latent confounder, isolated feature and latent mediator |
| Labels vs. LOCO | Relevant features have higher LOCO (AUC 0.70) |
| Prediction pipeline | Runs on all 3 test datasets; checkpoint round trip works |

---

## History

**v1 (commit `3b1523e`):** nanoTabPFN with a row-collapse head, trained on LOCO. It learned nothing: Spearman 0.02, worse than a constant guess.

| v1 problem | Status now |
|---|---|
| Data loader always served the same 32 tasks | Fixed: `PriorTasks` with shuffle and held-out split |
| `SmoothL1Loss` mean reduction made the mask a no-op | Fixed: per-task losses, no padding |
| Attention pooling computed and then discarded | Replaced by the CLS-attention readout |
| Prediction split N−1 with all labels visible | Fixed: inference uses all rows as context, by design |
| Categorical columns coerced to NaN → 0 in prediction | Fixed: TabICL preprocessing |
| LOCO targets mostly noise | **Replaced** by generator relevance labels (noise-free) and the label-free gating objective |

**Why LOCO was dropped as the default.**
- **Noise:** re-running the same LOCO with another seed gives Spearman 0.22 (v1 prior). In the v2 prior the std across 3 repeats (0.13) is close to the std of the targets themselves (0.19).
- **Redundancy:** two correlated, useful features each get LOCO ≈ 0.
- **Teacher bias:** the targets inherit the Random Forest's biases.
- **Mixed scales:** binary tasks use ΔAUC, multiclass tasks Δ(−log-loss).
- **Cost:** it is ~15× slower to build than the generator labels.

## Branches

| Branch | Notes |
|---|---|
| `test_2` | **This work.** v2 is uncommitted on top of `3b1523e` (v1), which is the same commit as `test` / `origin/test` |
| `main` (`cd80534`) | Stubs out the LOCO loop as a `TODO`, which looks like an internship exercise. **Don't blindly merge `test_2` → `main`** |
| `MetaTabFS` | Earlier metafeature approach; `data/prior/new_prior.h5` belongs to it |
