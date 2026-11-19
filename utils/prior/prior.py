"""Feature-selection prior: synthetic tasks from TabICL's SCM prior + per-feature targets.

Kumo Tabular's own data generator is not released; NVIDIA describes its training as similar to
TabICLv2, whose SCM prior ships with the ``tabicl`` package. Tasks are generated with
``utils.prior.relevance.RelevanceSCMPrior`` (identical tasks to ``tabicl.prior``) which also returns
ground-truth relevance labels from the generating causal graph (see relevance.py).

Targets stored per feature:
    relevance               always   1 = feature is in y's Markov boundary (not d-separated from y given
                                     the other features), 0 = irrelevant; from the generator, noise-free
    feature_importance      optional LOCO = score(all) - score(all but feature), mean over repeats
    feature_importance_std  optional LOCO std over repeats

HDF5 layout (N tasks, padded to T_max rows and H_max features):
    X (N, T_max, H_max) float32 standardized by the prior, zero padded
    y (N, T_max) float32 class labels, -1 padded
    num_features, num_datapoints, train_size (N,) int
    relevance (N, H_max) int8, -1 padded
    feature_importance, feature_importance_std (N, H_max) float32, NaN padded (only if LOCO was computed)
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed

from utils.fs.fs import run_leave_one_feature_out

DEFAULT_PRIOR_KWARGS = dict(
    prior_type="mix_scm",  # MLP- and tree-based SCMs, as in TabICL
    min_features=3,
    max_features=20,
    max_classes=10,
    min_seq_len=100,
    max_seq_len=500,
    batch_size_per_gp=4,
)


def _loco_for_task(X: np.ndarray, y: np.ndarray, n_repeats: int) -> tuple[np.ndarray, np.ndarray]:
    result = run_leave_one_feature_out(X, y, n_repeats=n_repeats, n_jobs=1)
    return result["improvement"].to_numpy(np.float32), result["improvement_std"].to_numpy(np.float32)


def build_prior(
    output_path: str | Path,
    n_tasks: int,
    prior_kwargs: dict | None = None,
    n_loco_repeats: int = 0,
    chunk_size: int = 64,
    n_jobs: int = -1,
    seed: int = 0,
) -> None:
    """Generate ``n_tasks`` tasks with relevance labels; also LOCO targets if ``n_loco_repeats > 0``."""
    from tabicl.prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

    from utils.prior.relevance import RelevanceSCMPrior

    prior_kwargs = {**DEFAULT_PRIOR_KWARGS, **(prior_kwargs or {})}
    T_max, H_max = prior_kwargs["max_seq_len"], prior_kwargs["max_features"]
    with_loco = n_loco_repeats > 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = RelevanceSCMPrior(
        batch_size=chunk_size, fixed_hp=DEFAULT_FIXED_HP, sampled_hp=DEFAULT_SAMPLED_HP, **prior_kwargs
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with h5py.File(output_path, "w") as f:
        X_ds = f.create_dataset("X", (n_tasks, T_max, H_max), "float32", chunks=(1, T_max, H_max), compression="gzip")
        y_ds = f.create_dataset("y", (n_tasks, T_max), "float32", chunks=(1, T_max), compression="gzip")
        rel_ds = f.create_dataset("relevance", (n_tasks, H_max), "int8")
        nf_ds = f.create_dataset("num_features", (n_tasks,), "int64")
        nd_ds = f.create_dataset("num_datapoints", (n_tasks,), "int64")
        ts_ds = f.create_dataset("train_size", (n_tasks,), "int64")
        if with_loco:
            imp_ds = f.create_dataset("feature_importance", (n_tasks, H_max), "float32")
            std_ds = f.create_dataset("feature_importance_std", (n_tasks, H_max), "float32")
        f.attrs["prior_kwargs"] = json.dumps(prior_kwargs)
        f.attrs["relevance_definition"] = (
            "1 iff the feature is not d-separated from y given all other observed features in the "
            "generating SCM graph (strong relevance / Markov boundary)"
        )
        if with_loco:
            f.attrs["loco_definition"] = (
                f"score(all) - score(all but feature), mean over {n_loco_repeats} random 70/30 splits of all "
                "rows; Random Forest teacher; binary: ROC-AUC, multiclass: -log-loss"
            )

        written = 0
        while written < n_tasks:
            X, y, d, seq_lens, train_sizes, relevance = prior.get_batch_with_relevance()
            tasks = []
            for i in range(len(d)):
                H, T = int(d[i]), int(seq_lens[i])
                Xi, yi = X[i, :T, :H].numpy(), y[i, :T].numpy()
                if len(np.unique(yi)) < 2:
                    continue  # degenerate task
                tasks.append((Xi, yi, int(train_sizes[i]), relevance[i]))
            tasks = tasks[: n_tasks - written]

            if with_loco:
                loco = Parallel(n_jobs=n_jobs)(delayed(_loco_for_task)(Xi, yi, n_loco_repeats) for Xi, yi, _, _ in tasks)
            else:
                loco = [None] * len(tasks)

            for (Xi, yi, ts, rel), loco_i in zip(tasks, loco):
                T, H = Xi.shape
                X_pad = np.zeros((T_max, H_max), np.float32)
                X_pad[:T, :H] = Xi
                y_pad = np.full(T_max, -1, np.float32)
                y_pad[:T] = yi
                rel_pad = np.full(H_max, -1, np.int8)
                rel_pad[:H] = rel
                X_ds[written], y_ds[written], rel_ds[written] = X_pad, y_pad, rel_pad
                nf_ds[written], nd_ds[written], ts_ds[written] = H, T, ts
                if with_loco:
                    for ds, values in zip((imp_ds, std_ds), loco_i):
                        padded = np.full(H_max, np.nan, np.float32)
                        padded[:H] = values
                        ds[written] = padded
                written += 1

            print(f"prior: {written}/{n_tasks} tasks ({time.time() - start:.0f}s)")

    print(f"Saved prior with {n_tasks} tasks to {output_path}")


class PriorTasks:
    """In-memory view of a prior file that serves one task at a time (no padding)."""

    def __init__(self, path: str | Path, indices: np.ndarray | None = None):
        with h5py.File(path, "r") as f:
            self.X = f["X"][:]
            self.y = f["y"][:]
            self.relevance = f["relevance"][:] if "relevance" in f else None
            self.importance = f["feature_importance"][:] if "feature_importance" in f else None
            self.num_features = f["num_features"][:]
            self.num_datapoints = f["num_datapoints"][:]
            self.train_size = f["train_size"][:]
        self.path = path
        self.indices = np.arange(len(self.X)) if indices is None else np.asarray(indices)

    @property
    def has_relevance(self) -> bool:
        return self.relevance is not None

    @property
    def has_loco(self) -> bool:
        return self.importance is not None

    def __len__(self) -> int:
        return len(self.indices)

    def task(self, k: int) -> dict:
        i = self.indices[k]
        H, T = int(self.num_features[i]), int(self.num_datapoints[i])
        task = dict(
            X=torch.from_numpy(self.X[i, :T, :H]),
            y=torch.from_numpy(self.y[i, :T]),
            train_size=int(self.train_size[i]),
        )
        if self.has_relevance:
            task["relevance"] = torch.from_numpy(self.relevance[i, :H].astype(np.float32))
        if self.has_loco:
            task["importance"] = torch.from_numpy(self.importance[i, :H])
        return task

    def split(self, val_fraction: float, seed: int = 0) -> tuple["PriorTasks", "PriorTasks"]:
        perm = np.random.default_rng(seed).permutation(self.indices)
        n_val = max(1, int(round(val_fraction * len(perm))))
        return self._subset(perm[n_val:]), self._subset(perm[:n_val])

    def _subset(self, indices: np.ndarray) -> "PriorTasks":
        view = object.__new__(PriorTasks)
        view.__dict__.update(self.__dict__)
        view.indices = np.asarray(indices)
        return view

    def _values(self, key: str) -> np.ndarray:
        return np.concatenate([self.task(k)[key].numpy() for k in range(len(self))])

    def target_std(self) -> float:
        return float(self._values("importance").std())

    def relevance_rate(self) -> float:
        return float(self._values("relevance").mean())

    def iterate(self, tasks_per_step: int, seed: int = 0):
        """Infinite iterator over lists of tasks; reshuffles after every pass over the data."""
        tasks_per_step = min(tasks_per_step, len(self))
        rng = np.random.default_rng(seed)
        while True:
            order = rng.permutation(len(self))
            for start in range(0, len(order) - tasks_per_step + 1, tasks_per_step):
                yield [self.task(k) for k in order[start : start + tasks_per_step]]
