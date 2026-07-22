"""
make_dataset.py
Loads the raw Telco churn CSV, cleans it, and saves a processed version.
Run directly: python src/data/make_dataset.py
"""

import pandas as pd
from pathlib import Path
from src.config import DATA_VERSION , RAW_DATA_FILENAMES

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / RAW_DATA_FILENAMES[DATA_VERSION]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_cleaned_{DATA_VERSION}.csv"


def load_raw_data(path : Path) -> pd.DataFrame:
    """ load the raw csv file"""
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print("After to_numeric, missing:", df['TotalCharges'].isnull().sum())

    df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    print("After fillna, missing:", df['TotalCharges'].isnull().sum())

    df = df.drop_duplicates()
    df = df.drop(columns=['customerID'])
    return df

def save_processed_data(df : pd.DataFrame, path : Path) -> None:
    """ save the processed data to a csv file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    print("Loading raw data...")
    df = load_raw_data(RAW_DATA_PATH)
    print(f"Raw shape: {df.shape}")

    print("Cleaning data...")
    df_clean = clean_data(df)
    print(f"Cleaned shape: {df_clean.shape}")

    save_processed_data(df_clean, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()