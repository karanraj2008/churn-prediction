"""
make_dataset.py
Loads the raw Telco churn CSV, cleans it, and saves a processed version.
Run directly: python -m src.data.make_dataset
"""

import pandas as pd
from pathlib import Path
from src.config import DATA_VERSION, RAW_DATA_FILENAMES
from src.logger import get_logger

logger = get_logger(__name__)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / RAW_DATA_FILENAMES[DATA_VERSION]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_cleaned_{DATA_VERSION}.csv"

# Columns the rest of clean_data() assumes exist. If the raw file changes
# (new export, renamed column, etc.), fail here with a clear message rather
# than a confusing KeyError three functions deep.
REQUIRED_RAW_COLUMNS = [
    'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'
]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw csv file. Raises a clear error if it's missing, rather
    than pandas' generic FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {path}\n"
            f"Check that DATA_VERSION in src/config.py ('{DATA_VERSION}') "
            f"matches an entry in RAW_DATA_FILENAMES, and that the file "
            f"actually exists in data/raw/."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_raw_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Fail loudly, before any cleaning happens, if an expected column is
    missing — e.g. because a new raw data export renamed something."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing expected column(s): {missing}. "
            f"clean_data() assumes these exist — check the raw CSV's headers."
        )
    logger.info("Raw data validation passed: all required columns present")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    logger.info(f"After to_numeric, missing: {df['TotalCharges'].isnull().sum()}")

    df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    logger.info(f"After fillna, missing: {df['TotalCharges'].isnull().sum()}")

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Dropping {n_dupes} duplicate rows")
    df = df.drop_duplicates()

    df = df.drop(columns=['customerID'])
    return df


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Fail loudly if cleaning didn't fully work, rather than silently
    saving a file with missing values for the next stage to trip over."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        raise ValueError(
            f"Cleaned data still has missing values, cleaning did not fully "
            f"work:\n{missing}"
        )
    if len(df) == 0:
        raise ValueError("Cleaned data has 0 rows — check upstream filtering/dropping logic.")
    logger.info("Cleaned data validation passed: no missing values, non-empty")


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save the processed data to a csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to: {path}")


def main():
    df = load_raw_data(RAW_DATA_PATH)
    validate_raw_columns(df, REQUIRED_RAW_COLUMNS)

    df_clean = clean_data(df)
    validate_cleaned_data(df_clean)
    logger.info(f"Cleaned shape: {df_clean.shape}")

    save_processed_data(df_clean, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()