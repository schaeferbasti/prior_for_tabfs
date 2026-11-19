"""Prediction pipeline for TabFS v2: one forward pass per dataset -> one importance score per feature."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tabicl._sklearn.preprocessing import PreprocessingPipeline, TransformToNumerical  # noqa: E402

from utils.tfm.tabicl_fs import get_default_device, load_tabicl_fs  # noqa: E402

DATASETS = {  # file name in data/test_data -> target column
    "test_dataset_1": "area",
    "test_dataset_2": "disease",
    "test_dataset_3": "HasTumor",
}
# "latest" = most recent utils/tfm/tabicl_fs_*.pt, a path = that checkpoint,
# None = pretrained TabICLv2 with an untrained readout (zero-shot CLS attention, as a baseline;
#        scores are then read as relevance logits)
CHECKPOINT = "latest"
RESULTS_DIR = ROOT / "data/test_results"


def resolve_checkpoint(checkpoint):
    if checkpoint != "latest":
        return checkpoint
    candidates = sorted((ROOT / "utils/tfm").glob("tabicl_fs_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No utils/tfm/tabicl_fs_*.pt found. Run the pretraining pipeline first, "
                                "or set CHECKPOINT = None for the zero-shot baseline.")
    return candidates[-1]


def prepare(df: pd.DataFrame, target: str, max_classes: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Same preprocessing as TabICL's sklearn wrapper (one ensemble member, no normalization method)."""
    X, y = df.drop(columns=[target]), df[target]
    X_num = TransformToNumerical().fit_transform(X)  # categoricals -> ordinal codes, NaN handling
    X_num = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit_transform(X_num)

    # The backbone is a classifier: continuous targets are binned into at most max_classes quantile bins.
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > max_classes:
        y = pd.qcut(y, q=max_classes, labels=False, duplicates="drop")
    y = LabelEncoder().fit_transform(y.astype(str) if not pd.api.types.is_numeric_dtype(y) else y)
    return X_num.astype(np.float32), y.astype(np.float32), list(X.columns)


def main():
    checkpoint = resolve_checkpoint(CHECKPOINT)
    print(f"Checkpoint: {checkpoint or 'pretrained TabICLv2 (untrained readout)'}")
    model = load_tabicl_fs(checkpoint)
    device = get_default_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, target in DATASETS.items():
        df = pd.read_parquet(ROOT / "data/test_data" / f"{name}.parquet")
        X, y, columns = prepare(df, target, model.backbone.max_classes)
        score = model.feature_importance(X, y, device=device)
        value, selected = model.scores_to_selection(score)
        value_name = {"relevance": "p_relevant", "gating": "gate", "loco": "loco_importance"}[model.objective]

        result = (
            pd.DataFrame({"feature": columns, "score": score, value_name: value, "selected": selected})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        result["rank"] = np.arange(1, len(result) + 1)
        result.to_parquet(RESULTS_DIR / f"{name}_results.parquet", index=False)

        print(f"\n{name}: {X.shape[0]} rows, {X.shape[1]} features, {len(np.unique(y))} classes "
              f"| objective {model.objective} | selected {int(selected.sum())}/{len(selected)}")
        print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
