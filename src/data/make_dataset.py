"""
make_dataset.py
Loads the raw Telco churn CSV, cleans it, and saves a processed version.
Run directly: python -m src.data.make_dataset

"""

import pandas as pd
from pathlib import Path
from src.config import DATA_VERSION, RAW_DATA_FILENAMES
from src.logger import get_logger

#log the data processing steps
logger = get_logger(__name__)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / RAW_DATA_FILENAMES[DATA_VERSION]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_cleaned_{DATA_VERSION}.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw csv file."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


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


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save the processed data to a csv file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to: {path}")


def main():
    df = load_raw_data(RAW_DATA_PATH)

    df_clean = clean_data(df)
    logger.info(f"Cleaned shape: {df_clean.shape}")

    save_processed_data(df_clean, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
