import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from utils.metafeature_extraction.metafeature_extractor import get_dataset_metafeatures
from utils.nanoTabPFN_utils import get_default_device
from utils.tfm.tfm import NanoTabPFNModel, NanoTabPFNClassifier


def main():
    # Get Data
    test_dataset = pd.read_parquet("../data/test_dataset.parquet")

    # Calculate Metafeatures
    task_id = "id_test_1"
    metafeatures_test_dataset = get_dataset_metafeatures(test_dataset, task_id)
    metafeatures_test_dataset.to_parquet("../data/metafeatures_test_dataset.parquet")

    # Predict with TabFS
    device = get_default_device()

    checkpoint = torch.load(
        "../utils/tfm/weights_30_09_2026.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    clf = NanoTabPFNClassifier(model, device)
    rng = np.random.default_rng(seed=42)
    y = pd.Series(
        rng.integers(0, 2, size=len(metafeatures_test_dataset)),
        index=metafeatures_test_dataset.index,
        name="improvement",
    ).to_numpy()

    X_df = metafeatures_test_dataset.drop(
        columns=["task_id", "feature", "feature - top"]
    )
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X = X_df.fillna(0.0).to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(pred)


if __name__ == "__main__":
    main()
