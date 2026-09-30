import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.nanoTabPFN_utils import get_default_device
from utils.tfm.tfm import NanoTabPFNModel, NanoTabPFNClassifier


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
        "../utils/tfm/weights_30_09_2026.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    clf = NanoTabPFNClassifier(model, device)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0).to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(pred)


if __name__ == "__main__":
    main()
