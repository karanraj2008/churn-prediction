import pandas as pd
from pathlib import Path


# ---- Paths ----
# Path(__file__) = this file's location. .parent.parent.parent walks up
# from src/data/make_dataset.py to the project root (churn-prediction/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "churn_cleaned.csv"


def load_raw_data(path : Path) -> pd.DataFrame:
    """ load the raw csv file"""
    df = pd.read_csv(path)
    return df

def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """ clean the raw data"""
    
    #1. Convert TotalCharges to numeric, coercing errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    #2. fill missing data in TotalCharges after calculation using tenure x MonthlyCharges
    df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"], inplace=True)
    
    #3. let's drop duplicate now 
    df.drop_duplicates(inplace=True)

    #4. drop customerID column as it doesn't affect the churn prediction
    df.drop(columns=["customerID"], inplace=True)

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