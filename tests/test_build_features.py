import pandas as pd
import pytest

from src.features.build_features import (
    collapse_redundant_categories,
    encoding_binary_columns,
    one_hot_encoding,
    build_feature_dataset,
    validate_cleaned_columns,
    validate_feature_dataset,
)


def make_sample_cleaned_df() -> pd.DataFrame:
    """Mimics make_dataset.py's output: TotalCharges already numeric,
    customerID already dropped."""
    return pd.DataFrame({
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "No internet service"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": [29.85, 1936.3],
        "Churn": ["No", "Yes"],
    })


# --- collapse_redundant_categories() ---

def test_collapse_redundant_categories_removes_no_internet_service():
    df = make_sample_cleaned_df()
    result = collapse_redundant_categories(df)
    assert "No internet service" not in result["OnlineSecurity"].values


def test_collapse_redundant_categories_removes_no_phone_service():
    df = make_sample_cleaned_df()
    result = collapse_redundant_categories(df)
    assert "No phone service" not in result["MultipleLines"].values
    assert result.loc[0, "MultipleLines"] == "No"


# --- encoding_binary_columns() ---

def test_encoding_binary_columns_maps_yes_no_to_1_0():
    df = collapse_redundant_categories(make_sample_cleaned_df())
    result = encoding_binary_columns(df)
    assert set(result["Partner"].unique()) <= {0, 1}
    assert set(result["Churn"].unique()) <= {0, 1}


def test_encoding_binary_columns_maps_gender():
    df = collapse_redundant_categories(make_sample_cleaned_df())
    result = encoding_binary_columns(df)
    assert result.loc[0, "gender"] == 1  # Female -> 1
    assert result.loc[1, "gender"] == 0  # Male -> 0


# --- one_hot_encoding() ---

def test_one_hot_encoding_produces_numeric_columns_and_drops_first():
    df = make_sample_cleaned_df()
    result = one_hot_encoding(df)
    assert "InternetService" not in result.columns  # original column replaced
    # drop_first=True: with 2 rows we only see 2 categories, so 1 dummy column expected
    dummy_cols = [c for c in result.columns if c.startswith("InternetService_")]
    for col in dummy_cols:
        assert result[col].dtype == int


# --- build_feature_dataset() end-to-end ---

def test_build_feature_dataset_produces_fully_numeric_output():
    df = make_sample_cleaned_df()
    result = build_feature_dataset(df)
    non_numeric = result.select_dtypes(exclude=["number"]).columns.tolist()
    assert non_numeric == []


def test_build_feature_dataset_has_no_missing_values():
    df = make_sample_cleaned_df()
    result = build_feature_dataset(df)
    assert result.isnull().sum().sum() == 0


# --- validate_cleaned_columns() ---

def test_validate_cleaned_columns_passes_when_all_present():
    df = make_sample_cleaned_df()
    validate_cleaned_columns(df, ["gender", "Partner", "Churn"])  # should not raise


def test_validate_cleaned_columns_raises_when_missing():
    df = make_sample_cleaned_df().drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        validate_cleaned_columns(df, ["gender", "Contract"])


# --- validate_feature_dataset() ---

def test_validate_feature_dataset_passes_on_valid_output():
    df = build_feature_dataset(make_sample_cleaned_df())
    validate_feature_dataset(df)  # should not raise


def test_validate_feature_dataset_raises_on_non_numeric_column():
    df = build_feature_dataset(make_sample_cleaned_df())
    df["leftover_text_col"] = ["a", "b"]
    with pytest.raises(ValueError, match="non-numeric"):
        validate_feature_dataset(df)


def test_validate_feature_dataset_raises_on_missing_values():
    df = build_feature_dataset(make_sample_cleaned_df())
    df.loc[0, "tenure"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_feature_dataset(df)
