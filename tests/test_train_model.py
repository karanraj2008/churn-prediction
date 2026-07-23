import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.train_model import (
    validate_features_data,
    split_training_testing_data,
    scale_features,
    select_best_model,
    save_model,
    save_scaler,
    save_metrics,
)


def make_sample_features_df(n=100, seed=0) -> pd.DataFrame:
    """Synthetic model-ready data — mimics build_features.py's output."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "gender": rng.integers(0, 2, n),
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(18, 120, n),
        "TotalCharges": rng.uniform(18, 8000, n),
        "Contract_One year": rng.integers(0, 2, n),
        "Churn": rng.choice([0, 1], n, p=[0.735, 0.265]),
    })
    return df


# --- validate_features_data() ---

def test_validate_features_data_passes_on_valid_data():
    df = make_sample_features_df()
    validate_features_data(df, "Churn")  # should not raise


def test_validate_features_data_raises_if_target_missing():
    df = make_sample_features_df().drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Churn"):
        validate_features_data(df, "Churn")


def test_validate_features_data_raises_on_unexpected_target_values():
    df = make_sample_features_df()
    df.loc[0, "Churn"] = 2
    with pytest.raises(ValueError, match="unexpected values"):
        validate_features_data(df, "Churn")


def test_validate_features_data_raises_on_non_numeric_column():
    df = make_sample_features_df()
    df["leftover"] = "text"
    with pytest.raises(ValueError, match="non-numeric"):
        validate_features_data(df, "Churn")


def test_validate_features_data_raises_on_missing_values():
    df = make_sample_features_df()
    df.loc[0, "tenure"] = None
    with pytest.raises(ValueError, match="missing values"):
        validate_features_data(df, "Churn")


# --- split_training_testing_data() ---

def test_split_preserves_churn_ratio_via_stratify():
    df = make_sample_features_df(n=500)
    x_train, x_test, y_train, y_test = split_training_testing_data(df, "Churn", 0.2, 42)
    # stratified split should keep train/test churn rates close
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_split_is_reproducible_with_same_random_state():
    df = make_sample_features_df(n=200)
    x_train_1, x_test_1, _, _ = split_training_testing_data(df, "Churn", 0.2, 42)
    x_train_2, x_test_2, _, _ = split_training_testing_data(df, "Churn", 0.2, 42)
    assert x_train_1.equals(x_train_2)


def test_split_test_size_is_respected():
    df = make_sample_features_df(n=200)
    x_train, x_test, _, _ = split_training_testing_data(df, "Churn", 0.2, 42)
    assert len(x_test) == 40  # 20% of 200


# --- scale_features() ---

def test_scale_features_does_not_mutate_original_data():
    df = make_sample_features_df(n=100)
    x_train, x_test, y_train, y_test = split_training_testing_data(df, "Churn", 0.2, 42)
    x_train_before = x_train.copy()

    scale_features(x_train, x_test, ["tenure", "MonthlyCharges", "TotalCharges"])

    pd.testing.assert_frame_equal(x_train, x_train_before)  # original untouched


def test_scale_features_centers_training_data_near_zero_mean():
    df = make_sample_features_df(n=200)
    x_train, x_test, y_train, y_test = split_training_testing_data(df, "Churn", 0.2, 42)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test, numeric_cols)

    assert abs(x_train_scaled["tenure"].mean()) < 1e-6
    assert abs(x_train_scaled["tenure"].std() - 1.0) < 0.05


def test_scale_features_raises_if_column_missing():
    df = make_sample_features_df(n=50)
    x_train, x_test, y_train, y_test = split_training_testing_data(df, "Churn", 0.2, 42)
    with pytest.raises(ValueError, match="not found"):
        scale_features(x_train, x_test, ["tenure", "does_not_exist"])


# --- select_best_model() ---

def test_select_best_model_picks_highest_f1():
    results = [
        {"name": "A", "model": "model_a", "accuracy": 0.7, "churn_precision": 0.5, "churn_recall": 0.5, "churn_f1": 0.50},
        {"name": "B", "model": "model_b", "accuracy": 0.8, "churn_precision": 0.6, "churn_recall": 0.6, "churn_f1": 0.65},
        {"name": "C", "model": "model_c", "accuracy": 0.75, "churn_precision": 0.55, "churn_recall": 0.55, "churn_f1": 0.55},
    ]
    best = select_best_model(results)
    assert best["name"] == "B"


def test_select_best_model_raises_on_empty_results():
    with pytest.raises(ValueError, match="No model results"):
        select_best_model([])


# --- save_model / save_scaler / save_metrics (using tmp_path) ---

def test_save_model_creates_parent_dirs_and_file(tmp_path):
    fake_model = {"not": "a real model, just testing save/load"}
    path = tmp_path / "v1" / "model.pkl"
    save_model(fake_model, path)
    assert path.exists()


def test_save_metrics_writes_valid_json(tmp_path):
    import json
    results = [
        {"name": "A", "accuracy": 0.7, "churn_precision": 0.5, "churn_recall": 0.5, "churn_f1": 0.5},
    ]
    path = tmp_path / "v1" / "metrics.json"
    save_metrics(results, "A", "v1", path)

    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["selected_model"] == "A"
    assert data["data_version"] == "v1"
    assert len(data["models"]) == 1
