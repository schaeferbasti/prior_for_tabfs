import pandas as pd

from utils.build_nanotabpfn_prior import build_feature_metadata_prior
from utils.metafeature_extraction.metafeature_extractor import get_dataset_metadata

import random

import h5py
import numpy as np
import torch
from utils.tfm.tfm import NanoTabPFNModel, train
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_randomness_seed(0)

def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    if torch.cuda.is_available(): device = "cuda"
    return device

datasets = []
datasets.append(train_test_split(*load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0))


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(self, filename, num_steps, batch_size, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0
        if device is None:
            device = get_default_device()
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer:end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer:end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                )

    def __len__(self):
        return self.num_steps


def iter_prior_tasks(filename, max_tasks=None):
    """
    Yield individual, unpadded tasks from the HDF5 prior.

    Assumes the HDF5 layout used by your PriorDumpDataLoader:
    X[task, row, feature], y[task, row],
    num_features[task], num_datapoints[task],
    single_eval_pos[task].
    """
    with h5py.File(filename, "r") as f:
        total_tasks = f["X"].shape[0]
        if max_tasks is not None:
            total_tasks = min(total_tasks, max_tasks)

        for task_id in range(total_tasks):
            n_rows = int(f["num_datapoints"][task_id])
            n_features = int(f["num_features"][task_id])

            X = f["X"][task_id, :n_rows, :n_features]
            y = f["y"][task_id, :n_rows]

            columns = [f"f{j}" for j in range(n_features)]
            X_task = pd.DataFrame(X, columns=columns)

            yield (
                task_id,
                X_task.copy(),
                np.asarray(y),
            )


def evaluate_features(old_prior):
    return 1

def run_leave_one_feature_out(X_train):
    performance_column = pd.DataFrame(index=X_train.columns, columns=["improvement"])
    # Do exhaustive feature selection
    full_score = float(evaluate_features(X_train))
    for feature_name in X_train.columns:
        X_without_feature = X_train.drop(columns=feature_name)
        score_without_feature = float(evaluate_features(X_without_feature))
        improvement = full_score - score_without_feature
        performance_column.loc[feature_name, "improvement"] = improvement
    return performance_column

def main():
    device = get_default_device()
    filename = "../data/prior/300k_150x5_2.h5"
    old_prior = PriorDumpDataLoader(filename, num_steps=2500, batch_size=32, device=device)
    new_prior = []
    # Calculate Metafeatures
    for task_id, X_train, y_train in iter_prior_tasks(filename, max_tasks=10):
        metafeatures_old_prior_task = get_dataset_metadata(X_train, task_id)

        # Calc Exhaustive Feature Selection
        performance = run_leave_one_feature_out(X_train)

        # Save Result
        new_prior_task = metafeatures_old_prior_task.join(
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
    model, history = train(model, new_prior, lr=4e-3, steps_per_eval=25)
    print(model, history)


if __name__ == "__main__":
    main()