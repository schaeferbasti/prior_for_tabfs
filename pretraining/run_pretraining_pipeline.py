"""Pretraining (fine-tuning) pipeline for TabFS v2.

1. Build the prior: TabICL SCM tasks + ground-truth relevance labels from the generating graph
   (+ LOCO targets only for OBJECTIVE = "loco"). Skipped if the file exists and has what is needed.
2. Load pretrained TabICLv2 and fine-tune it so that the attention of one CLS token over the
   features becomes the feature-selection score (see utils/tfm/tabicl_fs.py and utils/tfm/train.py).
"""

import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.prior.prior import PriorTasks, build_prior  # noqa: E402
from utils.tfm.tabicl_fs import build_tabicl_fs  # noqa: E402
from utils.tfm.train import train  # noqa: E402

# ---- objective -----------------------------------------------------------------------------
# "relevance": supervised by generator labels (default)
# "gating":    self-supervised, CLS scores gate the features, TabICL must still predict y
# "loco":      supervised by LOCO importance (v1 target, kept for comparison; slow prior)
OBJECTIVE = "relevance"

# ---- prior ---------------------------------------------------------------------------------
PRIOR_PATH = ROOT / "data/prior/fs_prior_tabicl.h5"
REBUILD_PRIOR = False  # set True to regenerate the prior even if PRIOR_PATH exists
N_TASKS = 4000  # relevance labels only: ~2.5 min; with LOCO (3 repeats): ~30 min on 12 cores
N_LOCO_REPEATS = 3 if OBJECTIVE == "loco" else 0
PRIOR_KWARGS = dict(  # passed to tabicl's SCM prior
    prior_type="mix_scm",
    min_features=3,
    max_features=20,
    max_classes=10,
    min_seq_len=100,
    max_seq_len=500,
)

# ---- model / training ----------------------------------------------------------------------
IMPORTANCE_CLS = 0  # which of the 4 CLS tokens carries the feature-selection score
STEPS = 500  # ~0.25 s per task on CPU (gating: ~0.5 s) -> ~17 min (gating ~35 min) for 500 x 8 tasks
TASKS_PER_STEP = 8
LR_BACKBONE = 1e-5
LR_READOUT = 1e-2
LAMBDA_PRED = 0.1  # relevance / loco: weight of TabICL's CE that keeps the CLS tokens predictive
LAMBDA_L0 = 0.1  # gating: sparsity weight (higher = fewer features kept)
FREEZE_BACKBONE = False  # True = only train the readout (per-head weights + bias)
VAL_FRACTION = 0.1
CHECKPOINT_PATH = ROOT / f"utils/tfm/tabicl_fs_{OBJECTIVE}_{date.today():%d_%m_%Y}.pt"
SEED = 0


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_randomness_seed(SEED)

    needs_rebuild = REBUILD_PRIOR or not PRIOR_PATH.exists()
    if not needs_rebuild and OBJECTIVE == "loco" and not PriorTasks(PRIOR_PATH).has_loco:
        print(f"{PRIOR_PATH} has no LOCO targets, rebuilding it.")
        needs_rebuild = True
    if needs_rebuild:
        build_prior(PRIOR_PATH, n_tasks=N_TASKS, prior_kwargs=PRIOR_KWARGS, n_loco_repeats=N_LOCO_REPEATS, seed=SEED)

    prior = PriorTasks(PRIOR_PATH)
    train_tasks, val_tasks = prior.split(VAL_FRACTION, seed=SEED)
    print(f"Prior: {len(train_tasks)} train / {len(val_tasks)} validation tasks | objective: {OBJECTIVE}")

    model = build_tabicl_fs(importance_cls=IMPORTANCE_CLS, objective=OBJECTIVE)
    train(
        model,
        train_tasks,
        val_tasks=val_tasks,
        steps=STEPS,
        tasks_per_step=TASKS_PER_STEP,
        lr_backbone=LR_BACKBONE,
        lr_readout=LR_READOUT,
        lambda_pred=LAMBDA_PRED,
        lambda_l0=LAMBDA_L0,
        freeze_backbone=FREEZE_BACKBONE,
        checkpoint_path=CHECKPOINT_PATH,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
