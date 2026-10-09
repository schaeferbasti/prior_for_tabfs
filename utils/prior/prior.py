from pathlib import Path

import h5py
import numpy as np

from utils.tfm.nanoTabPFN_utils import iter_prior_tasks
from utils.fs.fs import run_leave_one_feature_out


def build_loco_prior(
    old_prior_path: str,
    output_path: str,
    max_tasks: int | None = None,
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
