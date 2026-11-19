"""Leave-one-column-out (LOCO) teacher that produces the feature-importance targets of the prior."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


def get_task_type(y) -> str:
    unique_vals = np.unique(y)
    if len(unique_vals) == 2:
        return "binary"
    if len(unique_vals) <= 20 and np.all(np.mod(unique_vals, 1) == 0):
        return "multiclass"
    return "regression"


def _split(X, y, task_type, seed):
    stratify = y if task_type != "regression" else None
    try:
        return train_test_split(X, y, train_size=0.7, random_state=seed, stratify=stratify)
    except ValueError:  # a class with a single sample cannot be stratified
        return train_test_split(X, y, train_size=0.7, random_state=seed)


def evaluate_data(X, y, seed: int = 0, n_jobs: int = 1) -> float:
    """Score of a Random Forest on one 70/30 split (higher is better).

    binary -> ROC-AUC, multiclass -> -log-loss, regression -> -RMSE.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    task_type = get_task_type(y)
    X_train, X_val, y_train, y_val = _split(X, y, task_type, seed)

    if task_type == "regression":
        rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        return -root_mean_squared_error(y_val, rf.predict(X_val))

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs)
    rf.fit(X_train, y_train)

    # Probabilities over ALL classes of the task, so classes missing from the train split don't crash.
    labels = np.unique(y)
    proba = np.full((len(X_val), len(labels)), 1e-6)
    proba[:, np.searchsorted(labels, rf.classes_)] = rf.predict_proba(X_val)
    proba /= proba.sum(axis=1, keepdims=True)

    if task_type == "binary":
        if len(np.unique(y_val)) < 2:  # AUC undefined; fall back to log-loss on this split
            return -float(log_loss(y_val, proba, labels=labels))
        return float(roc_auc_score(y_val, proba[:, 1]))
    return -float(log_loss(y_val, proba, labels=labels))


def run_leave_one_feature_out(X, y, n_repeats: int = 1, n_jobs: int = 1) -> pd.DataFrame:
    """Leave-one-feature-out importance, averaged over ``n_repeats`` random splits.

    Returns a DataFrame indexed by feature with columns
      - 'improvement':     mean over repeats of  score(all features) - score(all but this feature)
                           (positive = removing the feature hurts = the feature is useful)
      - 'improvement_std': standard deviation over repeats (a noise estimate for the target)
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=[f"f{j}" for j in range(np.asarray(X).shape[1])])
    y = np.asarray(y)

    deltas = np.zeros((n_repeats, X.shape[1]))
    for r in range(n_repeats):
        full_score = evaluate_data(X, y, seed=r, n_jobs=n_jobs)
        for j, feature_name in enumerate(X.columns):
            deltas[r, j] = full_score - evaluate_data(X.drop(columns=feature_name), y, seed=r, n_jobs=n_jobs)

    return pd.DataFrame(
        {"improvement": deltas.mean(axis=0), "improvement_std": deltas.std(axis=0)},
        index=X.columns,
    )
