import random

import pandas as pd
import numpy as np
import torch

from utils.nanoTabPFN_utils import PriorDumpDataLoader, get_default_device
from utils.tfm.tfm import NanoTabPFNModel, train


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_data(data):
    # TODO: Implement a proper evaluation function that fits and predicts a model and returns a performance score for the given data.
    return random.random()


def run_leave_one_feature_out(X_train):
    """ Run leave-one-feature-out feature selection
    Return: pd.DataFrame with one column called 'improvement', each row represents the respective feature and the value is the improvement in performance when that feature is removed."""
    performance_column = pd.DataFrame(index=X_train.columns, columns=["improvement"])
    full_score = float(evaluate_data(X_train))
    for feature_name in X_train.columns:
        X_without_feature = X_train.drop(columns=feature_name)
        score_without_feature = float(evaluate_data(X_without_feature))
        improvement = full_score - score_without_feature
        performance_column.loc[feature_name, "improvement"] = improvement
    return performance_column


def main():
    old_prior_path = "../data/prior/300k_150x5_2.h5"
    old_prior = PriorDumpDataLoader(
        filename=old_prior_path,
        num_steps=2500,
        batch_size=32,
        device=get_default_device(),
    )
    # TODO: Build a new prior based on the old one adding leave-one-feature-out feature selection results
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    model, history = train(model, old_prior, lr=4e-3, steps_per_eval=25, checkpoint_path="../utils/tfm/weights_01_10_2026.pt")
    print(model, history)


if __name__ == "__main__":
    main()
