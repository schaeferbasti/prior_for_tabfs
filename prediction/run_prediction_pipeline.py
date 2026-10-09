import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from utils.tfm.nanoTabPFN_utils import get_default_device
from utils.tfm.nanoTabPFN import NanoTabPFNModel


def main():
    name = "test_dataset_1"
    test_dataset = pd.read_parquet("../data/test_data/" + name + ".parquet")
    # test_dataset_1
    X, y = test_dataset.drop(columns=["area"]), test_dataset["area"]
    # test_dataset_2
    # X, y = test_dataset.drop(columns=["disease"]), test_dataset["disease"]
    # test_dataset_3
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
    X_np = X.fillna(0.0).to_numpy(dtype=np.float32)

    N, D = X_np.shape  # e.g. N rows, D=12 features

    # Prepare tensors: add batch dimension
    x = torch.from_numpy(X_np).float().unsqueeze(0).to(device)  # (1, N, D)
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
    X_columns = list(X.columns)

    df = pd.DataFrame({
        "feature": X_columns,
        "importance_raw": feature_importance,
    })
    df = df.sort_values("importance_raw", ascending=False).reset_index(drop=True)

    imp = df["importance_raw"].to_numpy()
    imp_shifted = imp - imp.min()
    if imp_shifted.max() > 0:
        imp_norm = imp_shifted / imp_shifted.max()
    else:
        imp_norm = np.zeros_like(imp_shifted)

    df["importance_normalized"] = imp_norm
    df.to_parquet("../data/test_results/" + name + "_results.parquet", index=False)
    np.set_printoptions(precision=12, floatmode="unique", suppress=True)
    print(df["importance_raw"].to_numpy())
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
