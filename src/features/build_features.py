"""
build_features.py
Loads the cleaned churn data and encodes it into a model-ready format.
Run directly: python -m src.features.build_features

Note: scaling is NOT done here. Scaling needs to be fit on the training
split only (to avoid data leakage), so it happens later in train_model.py,
right after the train/test split. Everything in this file is safe to apply
to the whole dataset because none of it depends on statistics calculated
from the data itself — it's just fixed rules (collapse this label, map
Yes/No to 1/0, one-hot encode these columns).
"""

import pandas as pd
from pathlib import Path
from src.config import DATA_VERSION
from src.logger import get_logger

#log the data processing steps
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_cleaned_{DATA_VERSION}.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_features_{DATA_VERSION}.csv"


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """Load the cleaned churn data."""
    df = pd.read_csv(path)
    logger.info(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def collapse_redundant_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    'No internet service' duplicates InternetService == 'No', and
    'No phone service' duplicates PhoneService == 'No'. Collapsing these
    to 'No' avoids a redundant category during one-hot encoding.
    """

    # collapse 'No internet service' to 'No'
    internet_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    for col in internet_cols:
        df[col] = df[col].replace({"No internet service": "No"})

    # collapse 'No phone service' to 'No'
    df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    return df


def encoding_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    return df


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    new_dummy_cols = [
        col for col in df.columns
        if any(col.startswith(prefix + '_') for prefix in cat_cols)]

    df[new_dummy_cols] = df[new_dummy_cols].astype(int)
    return df


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final feature dataset."""
    df = collapse_redundant_categories(df)
    df = encoding_binary_columns(df)
    df = one_hot_encoding(df)
    return df


def save_featured_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save the final feature dataset to a csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved featured dataset to: {path}")


def main():
    """Main function to build the feature dataset."""
    df = load_cleaned_data(CLEANED_DATA_PATH)
    df = build_feature_dataset(df)
    logger.info(f"Feature dataset shape: {df.shape}")
    save_featured_dataset(df, FEATURES_DATA_PATH)


if __name__ == "__main__":
    main()
