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

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_cleaned_{DATA_VERSION}.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_features_{DATA_VERSION}.csv"

# Columns the encoding functions below assume exist. If make_dataset.py's
# output ever changes shape, fail here with a clear message rather than a
# confusing KeyError inside encoding_binary_columns().
REQUIRED_CLEANED_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn',
    'InternetService', 'Contract', 'PaymentMethod',
]


def load_cleaned_data(path: Path) -> pd.DataFrame:
    """Load the cleaned churn data. Raises a clear error if it's missing —
    most likely cause: make_dataset.py hasn't been run yet for this version."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found at: {path}\n"
            f"Run 'python -m src.data.make_dataset' first to generate it."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_cleaned_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Fail loudly, before any encoding happens, if an expected column is
    missing from the cleaned data."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Cleaned data is missing expected column(s): {missing}. "
            f"The encoding functions in this file assume these exist."
        )
    logger.info("Cleaned data validation passed: all required columns present")


def collapse_redundant_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    'No internet service' duplicates InternetService == 'No', and
    'No phone service' duplicates PhoneService == 'No'. Collapsing these
    to 'No' avoids a redundant category during one-hot encoding.
    """
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


def validate_feature_dataset(df: pd.DataFrame) -> None:
    """Fail loudly if encoding left behind anything unexpected — e.g. a
    Yes/No mapping that silently produced NaN because of an unexpected
    category value, or a non-numeric column models can't consume."""
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Feature dataset has non-numeric column(s) after encoding: "
            f"{non_numeric_cols}. Every column should be numeric before "
            f"this data reaches train_model.py."
        )

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        raise ValueError(
            f"Feature dataset has missing values after encoding — likely an "
            f"unexpected category value that didn't match a mapping:\n{missing}"
        )
    logger.info("Feature dataset validation passed: all columns numeric, no missing values")


def save_featured_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save the final feature dataset to a csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved featured dataset to: {path}")


def main():
    """Main function to build the feature dataset."""
    df = load_cleaned_data(CLEANED_DATA_PATH)
    validate_cleaned_columns(df, REQUIRED_CLEANED_COLUMNS)

    df = build_feature_dataset(df)
    validate_feature_dataset(df)
    logger.info(f"Feature dataset shape: {df.shape}")

    save_featured_dataset(df, FEATURES_DATA_PATH)


if __name__ == "__main__":
    main()