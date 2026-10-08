import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.nanoTabPFN_utils import get_default_device
from utils.tfm.nanoTabPFN import NanoTabPFNModel, NanoTabPFNClassifier


def main():
    test_dataset = pd.read_parquet("../data/test_dataset_1.parquet")
    X, y = test_dataset.drop(columns=["area"]), test_dataset["area"]
    # test_dataset = pd.read_parquet("../data/test_dataset_2.parquet")
    # X, y = test_dataset.drop(columns=["disease"]), test_dataset["disease"]
    # test_dataset = pd.read_parquet("../data/test_dataset_3.parquet")
    # X, y = test_dataset.drop(columns=["HasTumor"]), test_dataset["HasTumor"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    device = get_default_device()
    checkpoint = torch.load(
        "../utils/tfm/weights_08_10_2026.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0).to_numpy(dtype=np.float32)

    N, D = X.shape  # e.g. N rows, D=12 features

    # Prepare tensors: add batch dimension
    x = torch.from_numpy(X).float().unsqueeze(0).to(device)  # (1, N, D)
    y = torch.from_numpy(y).float().unsqueeze(0).to(device)  # (1, N)

    # All rows are “real”; all features are “real”
    row_mask = torch.ones((1, N), dtype=torch.bool, device=device)
    feature_mask = torch.ones((1, D), dtype=torch.bool, device=device)

    # Use at least one train and one test row for the internal TFM logic
    train_test_split_index = max(1, N - 1)

    with torch.no_grad():
        importance = model(
            (x, y),
            train_test_split_index=train_test_split_index,
            return_feature_importance=True,
            row_mask=row_mask,
            feature_mask=feature_mask,
        )  # shape: (1, D)

    feature_importance = importance[0].cpu().numpy()  # (D,)

    print("Feature importances:", feature_importance)
    print("Number of features:", len(feature_importance))


if __name__ == "__main__":
    main()
