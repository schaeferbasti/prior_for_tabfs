import numpy as np
import pandas as pd


def get_pandas_metafeature_names():
    return ['feature - count', 'feature - unique', 'feature - top', 'feature - freq', 'feature - mean', 'feature - std', 'feature - min', 'feature - 25%', 'feature - 50%', 'feature - 75%', 'feature - max']


def check_and_complete_pandas_description(feature_pandas_description):
    if "count" not in feature_pandas_description.index:
        feature_pandas_description.loc["count"] = np.nan
    if "unique" not in feature_pandas_description.index:
        feature_pandas_description.loc["unique"] = np.nan
    if "top" not in feature_pandas_description.index:
        feature_pandas_description.loc["top"] = np.nan
    if "freq" not in feature_pandas_description.index:
        feature_pandas_description.loc["freq"] = np.nan
    if "mean" not in feature_pandas_description.index:
        feature_pandas_description.loc["mean"] = np.nan
    if "std" not in feature_pandas_description.index:
        feature_pandas_description.loc["std"] = np.nan
    if "min" not in feature_pandas_description.index:
        feature_pandas_description.loc["min"] = np.nan
    if "25%" not in feature_pandas_description.index:
        feature_pandas_description.loc["25%"] = np.nan
    if "50%" not in feature_pandas_description.index:
        feature_pandas_description.loc["50%"] = np.nan
    if "75%" not in feature_pandas_description.index:
        feature_pandas_description.loc["75%"] = np.nan
    if "max" not in feature_pandas_description.index:
        feature_pandas_description.loc["max"] = np.nan
    return feature_pandas_description


def get_pandas_metafeatures(feature_df, featurename):
    feature_pandas_description = feature_df.describe(include="all")
    feature_pandas_description = check_and_complete_pandas_description(feature_pandas_description)
    feature_metadata = {
        "feature - count": feature_pandas_description.loc["count"].values[0],
        "feature - unique": feature_pandas_description.loc["unique"].values[0],
        "feature - top": feature_pandas_description.loc["top"].values[0],
        "feature - freq": feature_pandas_description.loc["freq"].values[0],
        "feature - mean": feature_pandas_description.loc["mean"].values[0],
        "feature - std": feature_pandas_description.loc["std"].values[0],
        "feature - min": feature_pandas_description.loc["min"].values[0],
        "feature - 25%": feature_pandas_description.loc["25%"].values[0],
        "feature - 50%": feature_pandas_description.loc["50%"].values[0],
        "feature - 75%": feature_pandas_description.loc["75%"].values[0],
        "feature - max": feature_pandas_description.loc["max"].values[0],
    }
    return feature_metadata


def get_pandas_metadata(X_train):
    columns = get_pandas_metafeature_names()
    pandas_metadata = pd.DataFrame(index=X_train.columns, columns=columns)
    for feature_name in X_train.columns:
        feature = pd.DataFrame(X_train[feature_name])
        feature_metadata = get_pandas_metafeatures(feature, feature_name)
        pandas_metadata.loc[feature_name] = [
            feature_metadata["feature - count"],
            feature_metadata["feature - unique"],
            feature_metadata["feature - top"],
            feature_metadata["feature - freq"],
            feature_metadata["feature - mean"],
            feature_metadata["feature - std"],
            feature_metadata["feature - min"],
            feature_metadata["feature - 25%"],
            feature_metadata["feature - 50%"],
            feature_metadata["feature - 75%"],
            feature_metadata["feature - max"],
        ]
    return pandas_metadata


def main():
    test_dataset = pd.read_parquet("../../data/test_dataset.parquet")
    metafeatures_test_dataset = get_pandas_metadata(test_dataset)
    metafeatures_test_dataset.to_parquet("../../data/metafeatures_test_dataset.parquet")


if __name__ == '__main__':
    main()