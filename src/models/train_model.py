"""
train_model.py
Loads model-ready features, splits, scales, trains Logistic Regression,
Random Forest, and XGBoost (the latter two tuned via GridSearchCV), then
saves the selected model + scaler + metrics for use in predict_model.py
and for comparing against future retrained versions.

Run directly: python -m src.models.train_model
"""

import json
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.config import DATA_VERSION
from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"churn_features_{DATA_VERSION}.csv"
MODEL_PATH = PROJECT_ROOT / "models" / DATA_VERSION / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / DATA_VERSION / "scaler.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / DATA_VERSION / "metrics.json"

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
TARGET_COL = 'Churn'
TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_PARAM_GRID = {
    'max_depth': list(range(1, 20)),
    'min_samples_leaf': list(range(1, 20)),
}
XGB_PARAM_GRID = {
    'max_depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
}


def load_features_data(path: Path) -> pd.DataFrame:
    """Load the model-ready output of build_features.py."""
    df = pd.read_csv(path)
    logger.info(f"Loaded features data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_training_testing_data(df: pd.DataFrame, target_column: str,
                                 test_size: float, random_state: int):
    """
    Split into train/test BEFORE any fitting happens (scaling, SMOTE, etc.
    would all leak information if fit before this split).
    stratify=y keeps the churn ratio consistent across train and test.
    """
    x = df.drop(columns=[target_column])
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Split data: x_train={x_train.shape}, x_test={x_test.shape}, "
        f"train churn rate={y_train.mean():.3f}, test churn rate={y_test.mean():.3f}"
    )
    return x_train, x_test, y_train, y_test


def scale_features(x_train, x_test, numeric_cols):
    """
    Fit StandardScaler on x_train only, apply (transform, not fit_transform)
    to x_test. Fitting on the full dataset before splitting would leak test
    set statistics into training.
    """
    scaler = StandardScaler()

    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[numeric_cols] = scaler.fit_transform(x_train_scaled[numeric_cols])
    x_test_scaled[numeric_cols] = scaler.transform(x_test_scaled[numeric_cols])

    logger.info(f"Scaled {len(numeric_cols)} numeric columns: {numeric_cols}")
    return x_train_scaled, x_test_scaled, scaler


