"""
Data loading, cleaning, target creation, feature engineering, and train/test preparation
for the Finland train-delay prediction project.

The script mirrors the main reusable preprocessing steps from
Finnish_dataset_EDA_and_Feature_Engg.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


SEED = 1

FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "type_encoded",
    "scheduled_time_minutes",
    "Air temperature",
    "Wind speed",
    "Snow depth",
    "Horizontal visibility",
    "sin_time",
    "cos_time",
    "sin_month",
    "cos_month",
    "sin_day",
    "cos_day",
]


def load_2025_data(data_dir: str | Path = "../data/raw") -> pd.DataFrame:
    """Load and combine the 12 monthly 2025 Parquet files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("matched_data_flat_2025_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No 2025 Parquet files found in {data_dir}")

    dfs = [pd.read_parquet(file) for file in files]
    return pd.concat(dfs, ignore_index=True)


def clean_and_engineer_features(data_2025: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's cleaning and feature-engineering rules."""
    selected_columns = [
        "departureDate",
        "trainType",
        "trainCategory",
        "stationName",
        "type",
        "scheduledTime",
        "differenceInMinutes",
        "Air temperature",
        "Wind speed",
        "Precipitation amount",
        "Snow depth",
        "Horizontal visibility",
    ]

    df = data_2025[selected_columns].copy()

    df["scheduledTime"] = pd.to_datetime(df["scheduledTime"])
    df["scheduled_time_minutes"] = (
        df["scheduledTime"].dt.hour * 60
        + df["scheduledTime"].dt.minute
    )

    df = df.dropna(subset=["differenceInMinutes"]).copy()

    # Positive differenceInMinutes means the train was delayed.
    df["delayed"] = (df["differenceInMinutes"] > 0).astype(int)

    # Required weather features.
    df = df.dropna(
        subset=["Wind speed", "Horizontal visibility"]
    ).copy()

    # Precipitation is excluded because it is highly incomplete.
    df = df.drop(columns=["Precipitation amount"])

    # In this dataset -1 is treated as a no-snow sentinel.
    df["Snow depth"] = df["Snow depth"].replace(-1, 0)
    df = df.dropna(subset=["Snow depth"]).copy()

    df["departureDate"] = pd.to_datetime(df["departureDate"])
    df["month"] = df["departureDate"].dt.month
    df["day_of_week"] = df["departureDate"].dt.dayofweek

    df["type_encoded"] = df["type"].map(
        {"DEPARTURE": 0, "ARRIVAL": 1}
    )

    if df["type_encoded"].isna().any():
        raise ValueError("Unexpected values found in 'type'.")

    # Station is outside the stated weather-focused modelling scope.
    df = df.drop(columns=["stationName"])

    # Cyclical time encoding.
    df["scheduled_time_minutes"] = pd.to_numeric(
        df["scheduled_time_minutes"], errors="coerce"
    )
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["day_of_week"] = pd.to_numeric(
        df["day_of_week"], errors="coerce"
    )

    df["sin_time"] = np.sin(
        2 * np.pi * df["scheduled_time_minutes"] / 1440
    )
    df["cos_time"] = np.cos(
        2 * np.pi * df["scheduled_time_minutes"] / 1440
    )
    df["sin_month"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["cos_month"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["sin_day"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["cos_day"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = SEED,
):
    """Split first, then undersample training data only."""
    X = df[FEATURE_COLUMNS]
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    rus = RandomUnderSampler(random_state=seed)
    X_train_resampled, y_train_resampled = rus.fit_resample(
        X_train, y_train
    )

    return (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
    )


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir: str | Path = "../data/processed",
) -> None:
    """Save the undersampled training set and untouched test set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    train_df["delayed"] = y_train

    test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
    test_df["delayed"] = y_test

    train_df.to_csv(output_dir / "train_undersampled.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    raw_dir = Path("../data/raw")
    processed_dir = Path("../data/processed")

    data = load_2025_data(raw_dir)
    prepared = clean_and_engineer_features(data)
    train_X, test_X, train_y, test_y = prepare_train_test(prepared)

    save_processed_data(
        train_X, test_X, train_y, test_y, processed_dir
    )

    print("Processed datasets created successfully.")
