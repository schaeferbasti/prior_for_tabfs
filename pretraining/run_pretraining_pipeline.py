import random
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from utils.nanoTabPFN_utils import PriorDumpDataLoader, get_default_device, iter_prior_tasks
from utils.tfm.nanoTabPFN import NanoTabPFNModel, train


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_data(X, y):
    unique_vals = np.unique(y)
    n_unique = len(unique_vals)
    if n_unique == 2:
        task_type = "binary"
    elif n_unique <= 20 and np.all(np.mod(unique_vals, 1) == 0):
        task_type = "multiclass"
    else:
        task_type = "regression"

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=0, stratify=y if len(np.unique(y)) <= 50 else None,
    )
    if task_type == "regression":
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=0,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        return -rmse

    else:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=0,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        if task_type == "binary":
            y_prob = rf.predict_proba(X_val)[:, 1]
            score = float(roc_auc_score(y_val, y_prob))
            return score
        else:
            y_prob = rf.predict_proba(X_val)
            ll = float(log_loss(y_val, y_prob))
            return ll


def run_leave_one_feature_out(X_train, y_train):
    """ Run leave-one-feature-out feature selection
    Return: pd.DataFrame with one column called 'improvement', each row represents the respective feature and the value is the improvement in performance when that feature is removed."""
    performance_column = pd.DataFrame(index=X_train.columns, columns=["improvement"])
    full_score = float(evaluate_data(X_train, y_train))
    for feature_name in X_train.columns:
        X_without_feature = X_train.drop(columns=feature_name)
        score_without_feature = float(evaluate_data(X_without_feature, y_train))
        improvement = full_score - score_without_feature
        performance_column.loc[feature_name, "improvement"] = improvement
    return performance_column


def build_loco_prior(
    old_prior_path: str,
    output_path: str,
    max_tasks: int | None = None,
    random_state: int = 0,
) -> None:
    """
    Build a new prior with LOCO feature-importance targets.

    For each task in the old prior:
      - Use the training portion (X_train, y_train).
      - Compute LOCO scores via run_leave_one_feature_out.
      - Store X, y, and feature_importance in a new HDF5 file.

    The new prior keeps the same task orientation as the old one:
      X: (T, N, D)
      y: (T, N)
      feature_importance: (T, D)
    where N and D are padded maxima.
    """
    # Collect all tasks in memory first (for simplicity).
    tasks = []

    for task_id, X_train, y_train in iter_prior_tasks(
        old_prior_path, max_tasks=max_tasks
    ):
        # LOCO scores: one per feature
        perf_df = run_leave_one_feature_out(X_train, y_train)
        importance = perf_df["improvement"].to_numpy(dtype=np.float32)

        # Full task data (for storing in the new prior)
        # Here we reuse the same training portion for both X and y.
        # If you later want to include validation rows too, adapt this.
        X_full = X_train.to_numpy(dtype=np.float32)
        y_full = y_train.astype(np.float32)

        tasks.append({
            "task_id": task_id,
            "X": X_full,          # (n_rows, n_cols)
            "y": y_full,          # (n_rows,)
            "importance": importance,  # (n_cols,)
        })

    if not tasks:
        raise ValueError("No tasks were collected from the old prior.")

    # Determine padded shapes
    max_rows = max(t["X"].shape[0] for t in tasks)
    max_cols = max(t["X"].shape[1] for t in tasks)
    n_tasks = len(tasks)

    # Allocate padded arrays
    X_all = np.full((n_tasks, max_rows, max_cols), np.nan, dtype=np.float32)
    y_all = np.full((n_tasks, max_rows), np.nan, dtype=np.float32)
    imp_all = np.full((n_tasks, max_cols), np.nan, dtype=np.float32)

    num_datapoints = np.zeros(n_tasks, dtype=np.int64)
    num_features = np.zeros(n_tasks, dtype=np.int64)
    single_eval_pos = np.zeros(n_tasks, dtype=np.int64)
    source_task_ids = np.zeros(n_tasks, dtype=np.int64)

    for i, t in enumerate(tasks):
        n_rows, n_cols = t["X"].shape
        X_all[i, :n_rows, :n_cols] = t["X"]
        y_all[i, :n_rows] = t["y"]
        imp_all[i, :n_cols] = t["importance"]

        num_datapoints[i] = n_rows
        num_features[i] = n_cols
        # Here we simply treat all rows as "training" for the student.
        # If you want a separate held-out portion, store that split here.
        single_eval_pos[i] = n_rows
        source_task_ids[i] = t["task_id"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "X",
            data=X_all,
            dtype="float32",
            chunks=(1, max_rows, max_cols),
            compression="gzip",
        )
        f.create_dataset(
            "y",
            data=y_all,
            dtype="float32",
            chunks=(1, max_rows),
            compression="gzip",
        )
        f.create_dataset(
            "feature_importance",
            data=imp_all,
            dtype="float32",
            chunks=(1, max_cols),
            compression="gzip",
        )
        f.create_dataset("num_datapoints", data=num_datapoints)
        f.create_dataset("num_features", data=num_features)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset(
            "max_num_classes",
            data=np.array([2], dtype=np.int64),  # placeholder
        )
        f.create_dataset("source_task_id", data=source_task_ids)

        f.attrs["target_definition"] = (
            "LOCO improvement = full_score - score_without_feature "
            "(Random Forest teacher, ROC AUC / -log_loss / -RMSE)"
        )

    print(
        f"Saved {n_tasks} tasks to {output_path}\n"
        f"X shape: {X_all.shape}\n"
        f"y shape: {y_all.shape}\n"
        f"feature_importance shape: {imp_all.shape}\n"
        f"Max rows: {max_rows}, max cols: {max_cols}"
    )


def main():
    old_prior_path = "../data/prior/300k_150x5_2.h5"
    new_prior_path = "../data/prior/loco_prior.h5"

    # Build the new prior with LOCO scores
    build_loco_prior(
        old_prior_path,
        new_prior_path,
        max_tasks=10,  # start small for testing
        random_state=0,
    )

    new_prior = PriorDumpDataLoader(
        filename=new_prior_path,
        num_steps=2500,
        batch_size=32,
        device=get_default_device(),
    )

    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2
    )
    model, history = train(model, new_prior, lr=4e-3, steps_per_eval=25, checkpoint_path="../utils/tfm/weights_08_10_2026.pt")
    print(model, history)


if __name__ == "__main__":
    main()
