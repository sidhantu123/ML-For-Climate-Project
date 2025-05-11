import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import os
from pathlib import Path


def create_cv_splits(data_path, output_dir, n_splits=3, test_size=0.2):
    """
    Create time series cross-validation splits for each city.

    Parameters:
    -----------
    data_path : str
        Path to the processed data file with lags
    output_dir : str
        Directory to save the CV splits
    n_splits : int
        Number of cross-validation folds
    test_size : float
        Size of test set relative to total data
    """
    # Read the processed data
    df = pd.read_csv(data_path)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique cities
    cities = df["city"].unique()

    for city in cities:
        print(f"Processing {city}...")

        # Filter data for current city
        city_data = df[df["city"] == city].copy()

        # Sort by date to ensure chronological order
        city_data["Date"] = pd.to_datetime(city_data["Date"])
        city_data = city_data.sort_values("Date")

        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(
            n_splits=n_splits, test_size=int(len(city_data) * test_size)
        )

        # Create city directory
        city_dir = output_path / city
        city_dir.mkdir(exist_ok=True)

        # Generate splits
        for fold, (train_idx, test_idx) in enumerate(tscv.split(city_data), 1):
            # Create fold directory
            fold_dir = city_dir / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            # Split data
            train_data = city_data.iloc[train_idx]
            test_data = city_data.iloc[test_idx]

            # Save splits
            train_data.to_csv(fold_dir / "train.csv", index=False)
            test_data.to_csv(fold_dir / "test.csv", index=False)

            print(
                f"  Created fold {fold} with {len(train_data)} training samples and {len(test_data)} test samples"
            )


if __name__ == "__main__":
    # Define paths
    data_path = "../data/processed/processed_data_with_lags.csv"
    output_dir = "../data/processed/cv_splits"

    # Create CV splits
    create_cv_splits(data_path, output_dir)
