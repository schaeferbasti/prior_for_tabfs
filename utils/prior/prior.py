"""Synthetic tasks with known feature relevance, as in the co-worker's study
`kumo-cls-studies/2026-11-13-encoder-importance-prior/run.py`.

Generator: TabICL v2's graph-SCM prior (`tabicl.prior.graph_lib`, the same code that kumo_prune vendors as
`_vendor/nanotabiclv2/prior.py`). Stage-1 shapes: 1024 rows, 1-100 features, train fraction 0.3-0.9;
50% regression, 50% classification (2-10 classes). Only the training rows are returned.

Ground truth per feature, from the DAG: relevant = the feature's node and the target's node share an ancestor
(this includes being the same node, an ancestor or a descendant); independent = no common ancestor.
This is looser than the Markov boundary: ~87% of features are relevant.

Differences from the original (not reproducible here):
- `kumo_prune` is private. Its `STAGE_CONFIG` is approximated by `STAGE_CONFIG` below ("filters on").
  Ask the co-worker for the exact config and replace it.
"""

from __future__ import annotations

import signal

import numpy as np
import torch

from tabicl.prior._dataset import SCMPrior, should_filter
from tabicl.prior._reg2cls import outlier_removing, standard_scaling
from tabicl.prior.graph_lib import _dataset as graph_dataset
from tabicl.prior.graph_lib._base import Context, DatasetProperties
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib._properties import sample_categorical_sizes

SEQ_LEN = 1024
MAX_FEATURES = 100
GEN_TIMEOUT_S = 20  # a few prior draws take minutes on the CPU; they are skipped (as in the original)

# Approximation of kumo_prune's STAGE_CONFIG: graph prior defaults with the dataset filters switched on.
STAGE_CONFIG = PriorConfig(
    filter_unpredictable_graphs=True,
    filter_unpredictable_datasets=True,
    remove_trivial_datasets=True,
)

# ---- record the DAG and the feature -> node assignment of the last sampled graph (same hook as the original)
_last: dict = {}
_orig_init = RandomGraphFunction.__init__


def _init(self, context, dag, node_feature_specs):
    _orig_init(self, context, dag, node_feature_specs)
    _last["dag"], _last["specs"] = dag, node_feature_specs


RandomGraphFunction.__init__ = _init  # patched on the class, so graph_lib's own reference sees it too


def ancestors(dag, node) -> set:
    """``dag[i]`` lists the parents of node i. Returns the node and all its ancestors."""
    seen, stack = {node}, [node]
    while stack:
        for p in dag[stack.pop()]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def sample_dataset(rng: np.random.Generator, config: PriorConfig = STAGE_CONFIG):
    """One task: (X_train (R, C) float, y_train (R,), relevant (C,) bool, regression bool)."""
    regression = bool(rng.random() < 0.5)
    train_size = int(rng.uniform(0.3, 0.9) * SEQ_LEN)
    num_features = round(rng.uniform(1, SCMPrior.adjust_max_features(SEQ_LEN, MAX_FEATURES)))
    num_classes = None if regression else int(rng.integers(2, 11))
    while True:
        context = Context(config=config)
        cat_sizes = {"x": sample_categorical_sizes(num_features, context, max_cat_size=200),
                     "y": [0 if regression else num_classes]}
        props = DatasetProperties(n_train=SEQ_LEN, n_test=0, cat_sizes=cat_sizes)
        ds = graph_dataset.RandomDataset(context).sample(props)
        dag, specs = _last["dag"], _last["specs"]
        data = ds.get_concat_tensors()
        names = [n for n, s in props.feature_specs.items() if s.group == "x" and s.cat_size > 1] + \
                [n for n, s in props.feature_specs.items() if s.group == "x" and s.cat_size <= 1]
        X = torch.cat([t for t in (data.get("x_cat"), data.get("x_num")) if t is not None], -1)
        if X.shape[-1] != len(names):
            continue
        X = standard_scaling(outlier_removing(X.float(), threshold=4))
        if regression:
            y = standard_scaling(outlier_removing(data["y_num"].float(), threshold=4)).view(-1)
        else:
            y = data["y_cat"].view(-1).long()
        if torch.isnan(X).any() or torch.isnan(y.float()).any():
            continue
        node_of = {n: i for i, s in enumerate(specs) for n in s}
        anc_y = ancestors(dag, node_of["y_0"])
        rel = np.array([bool(ancestors(dag, node_of[n]) & anc_y) for n in names])
        keep = (X[:train_size].std(0) > 1e-6).numpy()  # drop constant columns
        if keep.sum() < 2:
            continue
        X, rel = X[:, keep], rel[keep]
        if not regression and not SCMPrior.cls_sanity_check(X[None], y[None], train_size):
            continue
        if should_filter(X, y, config, is_classif=not regression):
            continue
        perm = rng.permutation(X.shape[1])
        return X[:train_size, perm], y[:train_size], rel[perm], regression


class _GenTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _GenTimeout


def iter_tasks(n: int, seed: int = 0):
    """Yield ``n`` tasks; draws slower than GEN_TIMEOUT_S are skipped. Seeds numpy/torch globally like the original."""
    signal.signal(signal.SIGALRM, _alarm)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    skipped = 0
    for _ in range(n):
        while True:
            signal.alarm(GEN_TIMEOUT_S)
            try:
                task = sample_dataset(rng)
                break
            except _GenTimeout:
                skipped += 1
            finally:
                signal.alarm(0)
        yield task, skipped
