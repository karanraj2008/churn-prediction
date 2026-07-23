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
    """Load the model-ready output of build_features.py. Raises a clear
    error if it's missing, rather than pandas' generic FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"Features data file not found at: {path}\n"
            f"Run 'python -m src.features.build_features' first to generate it."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded features data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_features_data(df: pd.DataFrame, target_column: str) -> None:
    """Fail loudly before training if the target column is missing, has
    unexpected values, or the dataset isn't fully numeric — training would
    otherwise fail deep inside sklearn with a much less obvious error."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in features data.")

    unexpected_values = set(df[target_column].unique()) - {0, 1}
    if unexpected_values:
        raise ValueError(
            f"Target column '{target_column}' has unexpected values: "
            f"{unexpected_values}. Expected only 0/1."
        )

    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Features data has non-numeric column(s): {non_numeric_cols}. "
            f"Check that build_features.py fully encoded these."
        )

    if df.isnull().sum().sum() > 0:
        raise ValueError("Features data contains missing values — cannot train on this.")

    logger.info("Features data validation passed")


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
    missing = [col for col in numeric_cols if col not in x_train.columns]
    if missing:
        raise ValueError(f"Cannot scale — column(s) not found in data: {missing}")

    scaler = StandardScaler()

    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[numeric_cols] = scaler.fit_transform(x_train_scaled[numeric_cols])
    x_test_scaled[numeric_cols] = scaler.transform(x_test_scaled[numeric_cols])

    logger.info(f"Scaled {len(numeric_cols)} numeric columns: {numeric_cols}")
    return x_train_scaled, x_test_scaled, scaler


def train_logistic_regression(x_train, y_train, random_state: int) -> LogisticRegression:
    """Baseline model. class_weight='balanced' handles the churn imbalance
    (73.5% No / 26.5% Yes) without duplicating any rows."""
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, param_grid: dict, random_state: int) -> RandomForestClassifier:
    """
    Tuned via GridSearchCV (5-fold CV on training data only, scoring F1 —
    not accuracy, since accuracy is misleading on imbalanced classes).
    Untuned Random Forest overfits badly (train ~99.8%, test ~79%) because
    trees grow unconstrained by default; tuning max_depth/min_samples_leaf
    fixes this.
    """
    logger.info("Training Random Forest (grid search)...")
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=random_state),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(x_train, y_train)
    logger.info(f"Random Forest best params: {grid.best_params_}")
    logger.info(f"Random Forest best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_xgboost(x_train, y_train, param_grid: dict, random_state: int) -> XGBClassifier:
    """
    Tuned via GridSearchCV, same reasoning as Random Forest.
    scale_pos_weight is XGBoost's equivalent of class_weight='balanced'.
    """
    logger.info("Training XGBoost (grid search)...")
    if (y_train == 1).sum() == 0:
        raise ValueError("y_train has no positive (churn=1) examples — cannot compute scale_pos_weight.")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    grid = GridSearchCV(
        XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric='logloss'),
        param_grid, cv=5, scoring='f1', n_jobs=-1
    )
    grid.fit(x_train, y_train)
    logger.info(f"XGBoost best params: {grid.best_params_}")
    logger.info(f"XGBoost best CV F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, x_test, y_test, name: str) -> dict:
    """Evaluate on the test set exactly once — never used during tuning."""
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(
        f"{name}: accuracy={report['accuracy']:.3f}  "
        f"churn_precision={report['1']['precision']:.3f}  "
        f"churn_recall={report['1']['recall']:.3f}  "
        f"churn_f1={report['1']['f1-score']:.3f}"
    )
    return {
        'name': name,
        'model': model,
        'accuracy': report['accuracy'],
        'churn_precision': report['1']['precision'],
        'churn_recall': report['1']['recall'],
        'churn_f1': report['1']['f1-score'],
    }


def select_best_model(results: list) -> dict:
    """
    Selects the model with the highest churn-class F1 as the primary model.
    NOTE: this optimizes for F1 (precision/recall balance), not recall alone.
    If the business priority is "never miss a churner over minimizing false
    alarms," a model with higher recall (even at lower F1) may be preferable
    — check the logged comparison and override this selection if so.
    """
    if not results:
        raise ValueError("No model results to select from.")
    best = max(results, key=lambda r: r['churn_f1'])
    logger.info(f"Selected model: {best['name']} (highest churn-class F1 = {best['churn_f1']:.3f})")
    return best


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to: {path}")


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to: {path}")


def save_metrics(results: list, best_name: str, data_version: str, path: Path) -> None:
    """
    Save every candidate model's metrics for this data version, so a future
    retrained version (v2, v3, ...) can be compared against this one without
    needing to re-run training just to check "did it actually improve?"
    """
    metrics = {
        'data_version': data_version,
        'selected_model': best_name,
        'models': [
            {
                'name': r['name'],
                'accuracy': r['accuracy'],
                'churn_precision': r['churn_precision'],
                'churn_recall': r['churn_recall'],
                'churn_f1': r['churn_f1'],
            }
            for r in results
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {path}")


def main():
    df = load_features_data(FEATURES_DATA_PATH)
    validate_features_data(df, TARGET_COL)

    x_train, x_test, y_train, y_test = split_training_testing_data(
        df, TARGET_COL, TEST_SIZE, RANDOM_STATE
    )
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test, NUMERIC_COLS)

    log_reg = train_logistic_regression(x_train_scaled, y_train, RANDOM_STATE)
    rf_model = train_random_forest(x_train_scaled, y_train, RF_PARAM_GRID, RANDOM_STATE)
    xgb_model = train_xgboost(x_train_scaled, y_train, XGB_PARAM_GRID, RANDOM_STATE)

    results = [
        evaluate_model(log_reg, x_test_scaled, y_test, "Logistic Regression"),
        evaluate_model(rf_model, x_test_scaled, y_test, "Random Forest (tuned)"),
        evaluate_model(xgb_model, x_test_scaled, y_test, "XGBoost (tuned)"),
    ]

    best = select_best_model(results)

    save_model(best['model'], MODEL_PATH)
    save_scaler(scaler, SCALER_PATH)
    save_metrics(results, best['name'], DATA_VERSION, METRICS_PATH)


if __name__ == "__main__":
    main()