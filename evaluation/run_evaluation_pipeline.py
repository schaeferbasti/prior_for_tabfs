"""Evaluate Kumo encoder-only feature importance on synthetic prior tasks with known relevance.

Port of `kumo-cls-studies/2026-11-13-encoder-importance-prior/run.py` + `analyse.py`.
- Tasks: `utils/prior/prior.py` (graph-SCM prior; relevant = shares an ancestor with y).
- Method: `utils/fs/kumo_fs.encoder_importance` (shuffle a column, relative change of Kumo-Large's row embedding).
- Baseline: |Spearman(x_j, y)|.
- Metric: per-task AUC (relevant vs independent), on tasks that have both kinds of feature.

Usage (from anywhere): python evaluation/run_evaluation_pipeline.py [worker_id] [n_tasks]
Each worker writes data/eval_results/worker<id>.jsonl (one line per task); the summary covers all worker files.
Reference (co-worker, 4380 tasks, 1414 scorable, CUDA): AUC encoder 0.839, |Spearman| 0.810.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.fs.kumo_fs import encoder_importance, get_device, load_network  # noqa: E402
from utils.prior.prior import iter_tasks  # noqa: E402

OUT = ROOT / "data/eval_results"


def run(worker: int, n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    nets = {True: load_network("regression", device=device), False: load_network("classification", device=device)}
    t_gen = t_enc = 0.0
    t0 = time.time()
    with open(OUT / f"worker{worker}.jsonl", "w") as fh:
        for i, ((X, y, rel, reg), skipped) in enumerate(iter_tasks(n, seed=1000 + worker)):
            t1 = time.time()
            imp = encoder_importance(nets[reg], X, y)
            t2 = time.time()
            t_gen += t1 - t0
            t_enc += t2 - t1
            Xn, yn = X.numpy(), y.float().numpy()
            corr = np.nan_to_num(np.array([abs(spearmanr(Xn[:, j], yn)[0]) if Xn[:, j].std() > 0 else 0.0
                                           for j in range(X.shape[1])]))
            r = {"id": f"{worker}-{i}", "regression": reg, "n_rows": int(X.shape[0]), "d": int(X.shape[1]),
                 "n_relevant": int(rel.sum()), "relevant": rel.astype(int).tolist(),
                 "imp_encoder": imp.round(6).tolist(), "imp_corr": corr.round(6).tolist()}
            if 0 < rel.sum() < len(rel):
                r["auc_encoder"] = roc_auc_score(rel, imp)
                r["auc_corr"] = roc_auc_score(rel, corr)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"worker {worker}: {i + 1}/{n} gen {t_gen / (i + 1):.2f}s enc {t_enc / (i + 1):.2f}s per task, "
                      f"{skipped} slow draws skipped", flush=True)
            t0 = time.time()


def _summary(g: pd.DataFrame) -> pd.Series:
    diff = g.auc_encoder - g.auc_corr
    boot = [np.random.default_rng(i).choice(diff, len(diff)).mean() for i in range(1000)]
    return pd.Series({"n": len(g), "auc_encoder": g.auc_encoder.mean(), "auc_corr": g.auc_corr.mean(),
                      "diff": diff.mean(), "diff_ci_lo": np.quantile(boot, 0.025),
                      "diff_ci_hi": np.quantile(boot, 0.975), "encoder_wins": (diff > 0).mean(),
                      "auc_encoder_ge_0.9": (g.auc_encoder >= 0.9).mean(),
                      "auc_corr_ge_0.9": (g.auc_corr >= 0.9).mean()})


def analyse() -> None:
    rows = [json.loads(line) for f in sorted(OUT.glob("worker*.jsonl")) for line in open(f)]
    df = pd.DataFrame(rows)
    if "auc_encoder" not in df:
        print(f"{len(df)} tasks, none with both relevant and independent features yet")
        return
    s = df.dropna(subset=["auc_encoder"]).copy()
    print(f"tasks: {len(df)}; with both relevant and independent features: {len(s)} ({len(s) / len(df):.0%}); "
          f"share of relevant features overall: {df.n_relevant.sum() / df.d.sum():.2f}")
    pd.set_option("display.width", 200)
    print("\n== all\n", _summary(s).round(3).to_string())
    s["task"] = np.where(s.regression, "regression", "classification")
    s["d_bin"] = pd.cut(s.d, [0, 10, 30, 60, 100])
    s["rows_bin"] = pd.cut(s.n_rows, [0, 400, 600, 800, 1024])
    for col in ["task", "d_bin", "rows_bin"]:
        print(f"\n== by {col}\n",
              s.groupby(col, observed=True).apply(_summary, include_groups=False).round(3).to_string())


if __name__ == "__main__":
    worker = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run(worker, n_tasks)
    analyse()
