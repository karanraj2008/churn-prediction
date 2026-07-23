import pandas as pd
import pytest

from src.data.make_dataset import (
    clean_data,
    validate_raw_columns,
    validate_cleaned_data,
)


def make_sample_raw_df() -> pd.DataFrame:
    """Mimics the real dataset's TotalCharges quirk: a blank string for a
    tenure == 0 customer."""
    return pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "tenure": [0, 5, 10],
        "MonthlyCharges": [29.85, 56.95, 42.30],
        "TotalCharges": [" ", "284.75", "423.00"],
        "Churn": ["No", "Yes", "No"],
    })


# --- clean_data() ---

def test_clean_data_fixes_total_charges_dtype_and_missing():
    df = make_sample_raw_df()
    result = clean_data(df)
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
    assert result["TotalCharges"].isna().sum() == 0
    assert result.loc[0, "TotalCharges"] == 0.0  # tenure=0 -> 0 * 29.85


def test_clean_data_drops_customer_id():
    df = make_sample_raw_df()
    result = clean_data(df)
    assert "customerID" not in result.columns


def test_clean_data_drops_duplicate_rows():
    df = make_sample_raw_df()
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = clean_data(df_with_dupe)
    assert len(result) == len(df)


def test_clean_data_leaves_unique_data_untouched():
    df = make_sample_raw_df()
    result = clean_data(df)
    assert len(result) == len(df)


# --- validate_raw_columns() ---

def test_validate_raw_columns_passes_when_all_present():
    df = make_sample_raw_df()
    validate_raw_columns(df, ["customerID", "tenure", "Churn"])  # should not raise


def test_validate_raw_columns_raises_when_column_missing():
    df = make_sample_raw_df().drop(columns=["tenure"])
    with pytest.raises(ValueError, match="tenure"):
        validate_raw_columns(df, ["customerID", "tenure", "Churn"])


# --- validate_cleaned_data() ---

def test_validate_cleaned_data_passes_on_clean_data():
    df = make_sample_raw_df()
    cleaned = clean_data(df)
    validate_cleaned_data(cleaned)  # should not raise


def test_validate_cleaned_data_raises_on_missing_values():
    df = make_sample_raw_df()
    cleaned = clean_data(df)
    cleaned.loc[0, "MonthlyCharges"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_cleaned_data(cleaned)


def test_validate_cleaned_data_raises_on_empty_dataframe():
    empty = pd.DataFrame(columns=["tenure", "MonthlyCharges", "TotalCharges"])
    with pytest.raises(ValueError, match="0 rows"):
        validate_cleaned_data(empty)
