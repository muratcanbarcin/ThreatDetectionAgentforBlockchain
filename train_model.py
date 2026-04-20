"""
Phase 1: train a Random Forest on the Ethereum fraud detection dataset.
Outputs: rf_model.pkl (pipeline: median imputation + RF), model_features.pkl (feature order).
"""

from __future__ import annotations

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
MODEL_PATH = ROOT / "rf_model.pkl"
FEATURES_PATH = ROOT / "model_features.pkl"


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Normalize column names (leading/trailing/double spaces)
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

    # Drop non-numeric ERC20 text columns (numeric ERC20 stats remain)
    erc20_text_cols = [
        c
        for c in X.columns
        if "erc20" in c.lower() and not pd.api.types.is_numeric_dtype(X[c])
    ]
    if erc20_text_cols:
        X = X.drop(columns=erc20_text_cols)

    # Numeric features only
    X = X.select_dtypes(include=[np.number])

    return X, y


def main() -> None:
    X, y = load_and_clean_data(DATA_PATH)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Missing values: median imputer fit on train only (no leakage)
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
    print(f"Accuracy Score: {acc:.6f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Feature list saved: {FEATURES_PATH} ({len(feature_names)} features)")


if __name__ == "__main__":
    main()
