"""
Catch Theft — Phase 1: train a Random Forest on the Ethereum fraud detection dataset.
Outputs under ``models/``: rf_model.pkl (pipeline: median imputation + RF), model_features.pkl (feature order).
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transaction_dataset.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURES_PATH = MODELS_DIR / "model_features.pkl"

logger = logging.getLogger(__name__)


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, normalize headers, drop id columns, and return numeric features with labels.

    Args:
        csv_path: Path to ``transaction_dataset.csv``.

    Returns:
        ``(X, y)`` where ``X`` is numeric features only and ``y`` is the ``FLAG`` column.

    Raises:
        KeyError: If ``FLAG`` is missing from the dataframe.
    """
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    drop_id_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in ("index", "address", "unnamed: 0") or c == "Unnamed: 0":
            drop_id_cols.append(c)
    df = df.drop(columns=drop_id_cols, errors="ignore")

    if "FLAG" not in df.columns:
        raise KeyError("Target column 'FLAG' not found in the dataset.")

    y = df["FLAG"]
    X = df.drop(columns=["FLAG"])

    erc20_text_cols = [
        c
        for c in X.columns
        if "erc20" in c.lower() and not pd.api.types.is_numeric_dtype(X[c])
    ]
    if erc20_text_cols:
        X = X.drop(columns=erc20_text_cols)

    X = X.select_dtypes(include=[np.number])

    return X, y


def main() -> None:
    """Train the pipeline, log metrics, and persist artifacts under ``models/``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_clean_data(DATA_PATH)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy Score: %.6f", acc)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred, digits=4))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    logger.info("Model saved: %s", MODEL_PATH)
    logger.info("Feature list saved: %s (%d features)", FEATURES_PATH, len(feature_names))


if __name__ == "__main__":
    main()
