import random

import pandas as pd
import numpy as np
import torch

from utils.nanoTabPFN_utils import build_feature_metadata_prior, get_default_device, PriorDumpDataLoader, iter_prior_tasks
from utils.tfm.tfm import NanoTabPFNModel, train


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_data(data):
    return random.random()


def run_leave_one_feature_out(X_train):
    performance_column = pd.DataFrame(index=X_train.columns, columns=["improvement"])
    # Do exhaustive feature selection
    full_score = float(evaluate_data(X_train))
    for feature_name in X_train.columns:
        X_without_feature = X_train.drop(columns=feature_name)
        score_without_feature = float(evaluate_data(X_without_feature))
        improvement = full_score - score_without_feature
        performance_column.loc[feature_name, "improvement"] = improvement
    return performance_column


def main():
    old_prior = "../data/prior/300k_150x5_2.h5"
    new_prior = []
    for task_id, X_train, y_train in iter_prior_tasks(old_prior, max_tasks=10):
        # Exhaustive Feature Selection
        performance = run_leave_one_feature_out(X_train)
        data = pd.concat(X_train, y_train, axis=1)
        new_prior_task = data.join(
            performance,
            how="left",
            validate="one_to_one",
        )
        new_prior.append(new_prior_task)
    new_prior = pd.concat(new_prior, ignore_index=True)
    build_feature_metadata_prior(
        feature_table=new_prior,
        output_path="../data/prior/new_prior.h5",
        target_mode="binary",
        test_fraction=0.2,
        random_state=0,
    )
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    new_prior = PriorDumpDataLoader("../data/prior/new_prior.h5", num_steps=2500,batch_size=32, device=get_default_device())
    model, history = train(model, new_prior, lr=4e-3, steps_per_eval=25, checkpoint_path="../utils/tfm/weights_30_09_2026.pt")
    print(model, history)


if __name__ == "__main__":
    main()
