"""Select features on the test datasets with Kumo's encoder (no training, no fitted model).

Port of `kumo-cls-studies/2026-11-13-encoder-select-catboost/select_features.py` (rule "noise")
and `select_v2.py` (rule "shadow", the default; it was the better one downstream with CatBoost).

Writes data/test_results/<name>_results.parquet: feature, importance, threshold, selected, rank.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.fs.kumo_fs import KumoEncoder, prepare_table, select_features  # noqa: E402

DATASETS = {  # file name in data/test_data -> (target column, classification?)
    "test_dataset_1": ("area", False),
    "test_dataset_2": ("disease", True),
    "test_dataset_3": ("HasTumor", True),
}
RULE = "shadow"  # "shadow" (v2) or "noise" (v1)
RESULTS_DIR = ROOT / "data/test_results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    encoders = {}
    for name, (target, classification) in DATASETS.items():
        df = pd.read_parquet(ROOT / "data/test_data" / f"{name}.parquet")
        X, y, scored, always = prepare_table(df.drop(columns=[target]), df[target], classification)
        task = "classification" if classification else "regression"
        if task not in encoders:
            encoders[task] = KumoEncoder(task)
        res = select_features(encoders[task], X, y, rule=RULE)

        result = pd.DataFrame({
            "feature": scored,
            "importance": [res["importance"][c] for c in scored],
            "threshold": [res["threshold"][c] for c in scored],
            "selected": [c in res["kept"] for c in scored],
        })
        result["margin"] = result.importance - result.threshold
        result = result.sort_values("margin", ascending=False).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        result.to_parquet(RESULTS_DIR / f"{name}_results.parquet", index=False)

        print(f"\n{name}: {len(X)} rows, {len(scored)} scored features ({len(always)} always kept: {always}) "
              f"| {task} | rule {RULE} | selected {int(result.selected.sum())}/{len(result)}")
        print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
