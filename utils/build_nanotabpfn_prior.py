from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def build_feature_metadata_prior(
    feature_table: pd.DataFrame,
    output_path: str | Path,
    *,
    target_mode: str = "binary",
    test_fraction: float = 0.2,
    random_state: int = 0,
) -> None:
    """
    Build a new TabPFN-compatible HDF5 prior.

    Each original dataset becomes one task:
        - rows/samples: original features
        - columns/features: metafeatures of an original feature
        - target: LOFO improvement or a derived label

    Parameters
    ----------
    feature_table:
        One row per (task_id, feature), containing metafeatures and
        an 'improvement' column.

    output_path:
        Destination HDF5 file.

    target_mode:
        'regression': raw continuous improvement values.
        'binary':     1 if improvement > 0, otherwise 0.
        'ternary':    0 harmful, 1 neutral, 2 helpful, determined from
                      within-task quantiles.

    test_fraction:
        Fraction of original features reserved as task-level test points.
        Each task must retain at least one train and one test feature.
    """
    required_columns = {"task_id", "feature", "improvement"}
    missing = required_columns - set(feature_table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    if feature_table.duplicated(["task_id", "feature"]).any():
        duplicates = feature_table.loc[
            feature_table.duplicated(["task_id", "feature"], keep=False),
            ["task_id", "feature"],
        ]
        raise ValueError(
            "Each (task_id, feature) must occur once. "
            f"Duplicate examples:\n{duplicates.head()}"
        )

    excluded_columns = {"task_id", "feature", "improvement"}
    metadata_columns = [
        col
        for col in feature_table.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(feature_table[col])
    ]

    if not metadata_columns:
        raise ValueError("No numeric metafeature columns were found.")

    data = feature_table.copy()
    data = data.dropna(subset=metadata_columns + ["improvement"]).copy()

    task_ids = sorted(data["task_id"].unique())
    if not task_ids:
        raise ValueError("No valid rows remain after dropping missing values.")

    grouped_tasks = []
    rng = np.random.default_rng(random_state)

    for task_id in task_ids:
        task_df = (
            data[data["task_id"] == task_id]
            .sort_values("feature")
            .reset_index(drop=True)
        )

        n_original_features = len(task_df)
        if n_original_features < 3:
            # You need enough features to have train and test feature rows.
            continue

        X_task = task_df[metadata_columns].to_numpy(dtype=np.float32)
        raw_improvement = task_df["improvement"].to_numpy(dtype=np.float32)

        if target_mode == "regression":
            y_task = raw_improvement

        elif target_mode == "binary":
            # Removing feature hurts performance -> feature is useful.
            y_task = (raw_improvement > 0).astype(np.int64)

        elif target_mode == "ternary":
            lower, upper = np.quantile(raw_improvement, [1 / 3, 2 / 3])

            y_task = np.ones(n_original_features, dtype=np.int64)
            y_task[raw_improvement <= lower] = 0
            y_task[raw_improvement >= upper] = 2

        else:
            raise ValueError(
                "target_mode must be one of: "
                "'regression', 'binary', 'ternary'."
            )

        # Randomly permute feature rows so that the first rows form the
        # context/train portion and the last rows form the query/test portion.
        permutation = rng.permutation(n_original_features)
        X_task = X_task[permutation]
        y_task = y_task[permutation]

        split_pos = int(np.floor((1 - test_fraction) * n_original_features))
        split_pos = min(max(split_pos, 1), n_original_features - 1)

        grouped_tasks.append({
            "task_id": task_id,
            "X": X_task,
            "y": y_task,
            "num_datapoints": n_original_features,
            "num_features": X_task.shape[1],
            "single_eval_pos": split_pos,
        })

    if not grouped_tasks:
        raise ValueError(
            "No tasks were usable. Each task needs at least 3 feature rows."
        )

    n_tasks = len(grouped_tasks)
    max_num_datapoints = max(task["num_datapoints"] for task in grouped_tasks)
    max_num_features = max(task["num_features"] for task in grouped_tasks)

    if target_mode == "binary":
        max_num_classes = 2
        y_dtype = np.int64
    elif target_mode == "ternary":
        max_num_classes = 3
        y_dtype = np.int64
    else:
        max_num_classes = 1
        y_dtype = np.float32

    X_all = np.full(
        (n_tasks, max_num_datapoints, max_num_features),
        fill_value=np.nan,
        dtype=np.float32,
    )

    # For classification, padded targets must still be valid class indices;
    # the loader must use num_datapoints to ignore padded positions.
    y_all = np.zeros(
        (n_tasks, max_num_datapoints),
        dtype=y_dtype,
    )

    num_datapoints = np.zeros(n_tasks, dtype=np.int64)
    num_features = np.zeros(n_tasks, dtype=np.int64)
    single_eval_pos = np.zeros(n_tasks, dtype=np.int64)
    source_task_ids = np.zeros(n_tasks, dtype=np.int64)

    for new_task_id, task in enumerate(grouped_tasks):
        n_rows = task["num_datapoints"]
        n_cols = task["num_features"]

        X_all[new_task_id, :n_rows, :n_cols] = task["X"]
        y_all[new_task_id, :n_rows] = task["y"]

        num_datapoints[new_task_id] = n_rows
        num_features[new_task_id] = n_cols
        single_eval_pos[new_task_id] = task["single_eval_pos"]
        source_task_ids[new_task_id] = task["task_id"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "X",
            data=X_all,
            compression="gzip",
            chunks=(1, max_num_datapoints, max_num_features),
        )
        f.create_dataset(
            "y",
            data=y_all,
            compression="gzip",
            chunks=(1, max_num_datapoints),
        )
        f.create_dataset("num_datapoints", data=num_datapoints)
        f.create_dataset("num_features", data=num_features)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset(
            "max_num_classes",
            data=np.asarray([max_num_classes], dtype=np.int64),
        )

        # Helpful provenance; ignored by your existing loader.
        f.create_dataset("source_task_id", data=source_task_ids)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "metafeature_names",
            data=np.asarray(metadata_columns, dtype=object),
            dtype=string_dtype,
        )
        f.attrs["target_mode"] = target_mode
        f.attrs["description"] = (
            "Feature-level prior: each task corresponds to an original "
            "tabular dataset; rows correspond to original features."
        )

    print(
        f"Saved {n_tasks} feature-selection tasks to {output_path}\n"
        f"X shape: {X_all.shape}\n"
        f"y shape: {y_all.shape}\n"
        f"Metafeatures per original feature: {max_num_features}\n"
        f"Target mode: {target_mode}"
    )