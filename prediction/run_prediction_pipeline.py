import pandas as pd

from utils.metafeature_extraction.metafeature_extractor import get_pandas_metadata


def main():
    # Get Data
    test_dataset = pd.read_parquet("../../data/test_dataset.parquet")

    # Calculate Metafeatures
    metafeatures_test_dataset = get_pandas_metadata(test_dataset)
    metafeatures_test_dataset.to_parquet("../../data/metafeatures_test_dataset.parquet")

    # Predict with TabFS

    # Save Result
