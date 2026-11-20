"""Feature importance from NVIDIA Kumo-Tabular's encoder, ported from the co-worker's studies
(`kumo-cls-studies`, 2026-11-13). No training: the pretrained weights are used as they are.

Importance of feature j = relative change of the encoder output when column j is shuffled across the rows:

    imp_j = mean_rows ||E(X with x_j shuffled) - E(X)|| / mean_rows ||E(X)||

E is Kumo's row embedding: the 4 readout ("CLS") tokens of every row, concatenated (4 x 256 = 1024 dims for
Large). Only the encoder runs (context rows with their labels); the ICL stage is never used.

Two entry points:
- `encoder_importance(net, X, y)`: tensors that are already preprocessed (prior tasks; the prior output is
  outlier-clipped and standardised, which is what Kumo's one-estimator recipe does).
  Source: `2026-11-13-encoder-importance-prior/run.py`.
- `KumoEncoder` + `select_features(...)`: pandas tables through Kumo's own recipe (one estimator). A feature is
  kept when it beats a null:
    rule="noise"  (v1): the stronger of two Gaussian noise columns added to the table.
                  Source: `2026-11-13-encoder-select-catboost/select_features.py`.
    rule="shadow" (v2, default, better downstream): a shuffled copy of the feature itself, re-scored the same way.
                  Source: `2026-11-13-encoder-select-catboost/select_v2.py`.

Kumo is loaded through NVIDIA's `sdm` package (structured-data-models, pinned commit 98f61289, as in TabArena).
The co-worker ran on CUDA with fp16 autocast; on CPU this runs in fp32, which is slower but otherwise identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import sdm

SIZE = "large"
CHUNK = 16  # shuffled copies per encoder call (prior path)
MAX_ROWS = 4096  # rows the real-data selector sees
BLOCK = 240  # scored columns per pass (Kumo takes at most 500 columns at once)
SHADOW_REPEATS = 2


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")


def load_estimator(task: str, size: str = SIZE, device: str | None = None) -> sdm.models.KumoTabular:
    """Pretrained Kumo-Tabular estimator for ``task`` ("classification" or "regression"), in eval mode."""
    return sdm.models.KumoTabular(task=task, size=size, pretrained=True, device=device or get_device()).eval()


def load_network(task: str, size: str = SIZE, device: str | None = None) -> torch.nn.Module:
    """The bare network (``.row_embedding`` is the encoder)."""
    return load_estimator(task, size, device).models[task]


# ----------------------------------------------------------------------------------------- prior tasks


@torch.no_grad()
def encoder_importance(net: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, chunk: int = CHUNK) -> np.ndarray:
    """X (R, C) preprocessed training rows, y (R,) their labels (long for classification). Returns (C,)."""
    device = next(net.parameters()).device
    R, C = X.shape
    x0 = X.to(device)
    yy = y.to(device)
    g = torch.Generator(device=device).manual_seed(0)
    copies = [x0]
    for j in range(C):
        xs = x0.clone()
        xs[:, j] = x0[torch.randperm(R, device=device, generator=g), j]
        copies.append(xs)
    embs = []
    with _autocast(device.type):
        for s in range(0, len(copies), chunk):
            xb = torch.stack(copies[s:s + chunk])
            B = xb.shape[0]
            e = net.row_embedding(xb, yy.expand(B, -1), torch.zeros(B, C, dtype=torch.bool, device=device))
            embs.append(e.float())
    E = torch.cat(embs)  # (C + 1, R, 1024)
    base = E[0]
    return ((E[1:] - base).norm(dim=-1).mean(-1) / base.norm(dim=-1).mean()).cpu().numpy()


# ------------------------------------------------------------------------------------------ real tables


class _Abort(Exception):
    pass


class KumoEncoder:
    """Kumo's recipe + encoder only, for pandas tables. The forward pass is aborted after the row embedding."""

    def __init__(self, task: str, size: str = SIZE, device: str | None = None):
        self.task = task
        self.device = device or get_device()
        self.est = load_estimator(task, size, self.device)
        self.est.models[task].row_embedding.register_forward_hook(self._abort)

    @staticmethod
    def _abort(module, inputs, output):
        raise _Abort(output.detach().float().clone())

    @torch.no_grad()
    def embed(self, X: pd.DataFrame, y: pd.Series) -> torch.Tensor:
        """Row embeddings (len(X), 1024) of the context rows X with labels y."""
        stypes = sdm.infer_stypes(X, _low_cardinality="infer")
        xc = sdm.TableTensor.from_pandas(df=X, stypes=stypes).to(self.device)
        xq = sdm.TableTensor.from_pandas(df=X.iloc[:1], stypes=stypes).to(self.device)
        yc = sdm.TableTensor.from_pandas(
            df=y.rename("y").to_frame(),
            stypes={"y": "categorical" if self.task == "classification" else "numerical"},
        ).to(self.device)
        try:
            with _autocast(self.device):
                self.est(x_context=xc, y_context=yc, x_query=xq, num_estimators=1,
                         generator=torch.Generator(self.device).manual_seed(0))
        except _Abort as a:
            E = a.args[0]
            return E.reshape(-1, E.shape[-2], E.shape[-1])[0][: len(X)]
        raise RuntimeError("encoder hook did not fire")


SCORED_DTYPES = ("float64", "float32", "int64", "int32", "int8", "int16", "uint8", "category", "bool", "boolean",
                 "Int64", "Float64")


def prepare_table(X: pd.DataFrame, y: pd.Series, classification: bool, max_rows: int = MAX_ROWS, seed: int = 0):
    """Subsample rows, split columns into scored / always kept, and encode y like the co-worker's selector."""
    idx = np.random.default_rng(seed).permutation(len(X))[:max_rows]
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)
    scored = [c for c in X.columns if str(X[c].dtype) in SCORED_DTYPES]
    always = [c for c in X.columns if c not in scored]  # text, dates, strings: kept, not scored
    X = X[scored].copy()
    for c in X.columns:  # bool -> category so Kumo reads it as a feature
        if str(X[c].dtype) in ("bool", "boolean"):
            X[c] = X[c].astype("category")
    if classification:
        y = pd.Series(pd.Categorical(y.astype(str)).codes, name="y")
    return X, y, scored, always


def _shuffled(df: pd.DataFrame, col, seed: int) -> pd.DataFrame:
    out = df.copy()
    out[col] = df[col].sample(frac=1, random_state=seed).set_axis(df.index)
    return out


def _score_block_noise(enc: KumoEncoder, X: pd.DataFrame, y: pd.Series, cols, rng):
    """v1: importance of every column and the threshold (the stronger of two noise columns)."""
    Xb = X[cols].copy()
    Xb["__noise1"] = rng.standard_normal(len(Xb))
    Xb["__noise2"] = rng.standard_normal(len(Xb))
    E0 = enc.embed(Xb, y)
    denom = E0.norm(dim=-1).mean()
    imp = {c: float((enc.embed(_shuffled(Xb, c, 1), y) - E0).norm(dim=-1).mean() / denom) for c in Xb.columns}
    thr = max(imp.pop("__noise1"), imp.pop("__noise2"))
    return imp, {c: thr for c in cols}


def _score_block_shadow(enc: KumoEncoder, X: pd.DataFrame, y: pd.Series, cols, repeats: int = SHADOW_REPEATS):
    """v2: real_j vs null_j, where null_j re-scores column j after first unlinking it from y by a shuffle."""
    Xb = X[cols].copy()

    def sens(base_df, E0, col, seed):
        return float((enc.embed(_shuffled(base_df, col, seed), y) - E0).norm(dim=-1).mean() / E0.norm(dim=-1).mean())

    E0 = enc.embed(Xb, y)
    real = {c: [] for c in cols}
    null = {c: [] for c in cols}
    for r in range(repeats):
        for c in cols:
            real[c].append(sens(Xb, E0, c, 1000 + r))
            Xn = _shuffled(Xb, c, 2000 + r)
            null[c].append(sens(Xn, enc.embed(Xn, y), c, 3000 + r))
    return {c: float(np.mean(real[c])) for c in cols}, {c: float(np.mean(null[c])) for c in cols}


def select_features(enc: KumoEncoder, X: pd.DataFrame, y: pd.Series, rule: str = "shadow", block: int = BLOCK,
                    seed: int = 0) -> dict:
    """Score every column of a (prepared) table and keep those above their null. Returns a result dict."""
    if rule not in ("shadow", "noise"):
        raise ValueError(f"rule must be 'shadow' or 'noise', got {rule!r}")
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    imp, thr = {}, {}
    for b in range(0, len(cols), block):
        part = cols[b:b + block]
        bi, bt = _score_block_shadow(enc, X, y, part) if rule == "shadow" else _score_block_noise(enc, X, y, part, rng)
        imp.update(bi)
        thr.update(bt)
    kept = [c for c in cols if imp[c] > thr[c]]
    if not kept and cols:  # always keep at least the best column
        kept = [max(cols, key=lambda c: imp[c] - thr[c])]
    return {"importance": imp, "threshold": thr, "kept": kept, "dropped": [c for c in cols if c not in kept]}
