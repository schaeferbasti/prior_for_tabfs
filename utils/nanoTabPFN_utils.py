import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def get_default_device():
    device = "cpu"
    #if torch.backends.mps.is_available(): device = "mps"
    #if torch.cuda.is_available(): device = "cuda"
    return device

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
        self.total_tasks = 0
        if device is None:
            device = get_default_device()
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                start = self.pointer
                end = start + self.batch_size
                if start >= self.total_tasks:
                    self.pointer = 0
                    start = 0
                    end = self.batch_size
                #end = self.pointer + self.batch_size
                num_features_batch = f["num_features"][start:end]
                num_datapoints_batch = f["num_datapoints"][start:end]
                max_num_features = int(num_features_batch.max())
                max_seq_in_batch = int(num_datapoints_batch.max())
                X_raw = f["X"][start:end, :max_seq_in_batch, :max_num_features]
                y_raw = f["y"][start:end, :max_seq_in_batch]
                # num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer:end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                imp_raw = f["feature_importance"][start:end, :max_num_features]
                x = torch.from_numpy(X_raw).to(torch.float32)
                y = torch.from_numpy(y_raw).to(torch.float32)
                feature_importance = torch.from_numpy(imp_raw).to(torch.float32)

                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer = end
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    train_test_split_index=train_test_split_index[0].item(),
                    feature_importance=feature_importance.to(self.device),
                )

    def __len__(self):
        return self.num_steps


def iter_prior_tasks(filename, max_tasks=None):
    """
    Yield individual tasks from the HDF5 prior.

    Relies on the HDF5 layout used by your PriorDumpDataLoader:
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