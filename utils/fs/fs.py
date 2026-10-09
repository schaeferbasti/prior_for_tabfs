import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

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
            return -ll


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
