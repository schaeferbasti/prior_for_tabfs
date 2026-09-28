import pandas as pd

from utils.metafeature_extraction.metafeature_extractor import get_pandas_metadata


def evaluate_features(old_prior):
    return 1

def run_exhaustive_feature_selection(old_prior):
    performance_column = pd.DataFrame(index=old_prior.columns, columns=["improvement"])
    # Do exhaustive feature selection
    full_score = float(evaluate_features(old_prior))
    for feature_name in old_prior.columns:
        X_without_feature = old_prior.drop(columns=feature_name)
        score_without_feature = float(evaluate_features(X_without_feature))
        improvement = full_score - score_without_feature
        performance_column.loc[feature_name, "improvement"] = improvement
    return performance_column

def main():
    # Get Old Prior
    old_prior = pd.read_parquet("../../data/old_prior.parquet")

    # Calculate Metafeatures
    metafeatures_old_prior = get_pandas_metadata(old_prior)

    # Calc Exhaustive Feature Selection
    performance_column = run_exhaustive_feature_selection(old_prior)

    # Save Result
    new_prior = pd.concat(metafeatures_old_prior, performance_column)
    new_prior.save_parquet("../../data/new_prior.parquet")
