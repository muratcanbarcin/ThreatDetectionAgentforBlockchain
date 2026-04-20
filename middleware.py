"""
Hybrid Detection Engine: GoPlus blacklist + Random Forest anomaly scoring + Groq advisory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import joblib
import pandas as pd
import requests
from groq import Groq

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "rf_model.pkl"
FEATURES_PATH = ROOT / "model_features.pkl"
GOPLUS_TEMPLATE = (
    "https://api.gopluslabs.io/api/v1/address_security/{address}?chain_id=1"
)


class ThreatDetectionAgent:
    def __init__(self) -> None:
        self._model = joblib.load(MODEL_PATH)
        self._feature_names: list[str] = list(joblib.load(FEATURES_PATH))

        api_key = os.environ.get("GROQ_API_KEY")
        self._groq: Groq | None = Groq(api_key=api_key) if api_key else None

    def fetch_goplus_security(self, address: str) -> tuple[bool, dict[str, Any]]:
        """Full GoPlus API JSON response and blacklist decision (for UI / logging)."""
        url = GOPLUS_TEMPLATE.format(address=address)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        except requests.RequestException as e:
            return False, {"error": str(e), "source": "goplus_http"}
        except ValueError as e:
            return False, {"error": str(e), "source": "goplus_json"}

        result = payload.get("result") or {}
        flags = (
            result.get("phishing_activities", "0"),
            result.get("malicious_behavior", "0"),
            result.get("stealing_attack", "0"),
        )
        threat = any(str(v) == "1" for v in flags)
        return threat, payload

    def check_blacklist(self, address: str) -> bool:
        threat, _payload = self.fetch_goplus_security(address)
        return threat

    def check_anomaly(self, features_dict: dict[str, Any]) -> bool:
        row = {
            name: float(features_dict.get(name, 0) or 0)
            for name in self._feature_names
        }
        df = pd.DataFrame([row], columns=self._feature_names)
        pred = int(self._model.predict(df)[0])
        return pred == 1

    def generate_llm_warning_detailed(
        self, address: str, threat_reason: str
    ) -> dict[str, Any]:
        prompt = (
            f"This wallet address ({address}) was flagged as risky because: {threat_reason}. "
            "Assess the situation and write a 2-3 sentence professional cybersecurity advisory telling "
            "the user to cancel the transaction if it still looks risky, or to approve if it does not. "
            "For each conclusion, give a confidence score from 0 to 100: "
            "0-20 very low; 21-40 low; 41-60 medium; 61-80 high; 81-100 very high. "
            "If the confidence score is 80 or above, approve the transaction; below 80, recommend cancellation."
        )

        if self._groq is None:
            text = (
                "Warning: GROQ_API_KEY is not set; LLM is disabled. "
                f"For address {address}, {threat_reason} was detected; canceling the transaction is recommended."
            )
            return {"content": text, "groq_raw": None, "used_llm": False}

        try:
            completion = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            msg = completion.choices[0].message
            text = (msg.content or "").strip()
            raw = (
                completion.model_dump()
                if hasattr(completion, "model_dump")
                else completion.dict()
            )
            return {"content": text, "groq_raw": raw, "used_llm": True}
        except Exception as e:
            text = (
                "Warning: Groq request failed (network or quota). "
                f"For address {address}, {threat_reason} was detected; canceling the transaction is recommended."
            )
            return {"content": text, "groq_raw": {"error": str(e)}, "used_llm": False}

    def generate_llm_warning(self, address: str, threat_reason: str) -> str:
        return str(self.generate_llm_warning_detailed(address, threat_reason)["content"])

    def evaluate_transaction(
        self, address: str, features_dict: dict[str, Any]
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        blacklisted = self.check_blacklist(address)
        anomalous = False if blacklisted else self.check_anomaly(features_dict)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if blacklisted or anomalous:
            if blacklisted:
                reason = (
                    "GoPlus blacklist / security flags "
                    "(phishing, malicious behavior, or theft risk)"
                )
            else:
                reason = "machine-learning-based anomaly score"

            llm_text = self.generate_llm_warning(address, reason)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "status": "Denied/Pending",
                "llm_warning": llm_text,
                "latency_ms": round(latency_ms, 2),
            }

        return {
            "status": "Allow",
            "llm_warning": None,
            "latency_ms": round(latency_ms, 2),
        }


def _fraud_profile_for_demo(
    feature_names: list[str], model: Any, max_tweaks: int = 12
) -> dict[str, float]:
    """Fill a zero mock vector from FLAG=1 sample columns until RF predicts class 1."""
    csv_path = ROOT / "data" / "transaction_dataset.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found for demo: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    drop_cols = [
        c
        for c in df.columns
        if c.lower() in ("index", "address", "unnamed: 0") or c == "Unnamed: 0"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    y = df["FLAG"]
    X = df.drop(columns=["FLAG"])
    erc20_text = [
        c
        for c in X.columns
        if "erc20" in c.lower() and not pd.api.types.is_numeric_dtype(X[c])
    ]
    if erc20_text:
        X = X.drop(columns=erc20_text)
    X = X.select_dtypes(include=["number"])
    X = X[feature_names]
    fraud_row = X.loc[y == 1].iloc[0]
    fraud_dict = {k: float(fraud_row[k]) for k in feature_names}

    by_magnitude = sorted(
        feature_names, key=lambda k: abs(fraud_dict[k]), reverse=True
    )

    base = {k: 0.0 for k in feature_names}
    n = min(max_tweaks, len(by_magnitude))
    while n <= len(by_magnitude):
        tweaked = dict(base)
        for k in by_magnitude[:n]:
            tweaked[k] = fraud_dict[k]
        df_try = pd.DataFrame([tweaked], columns=feature_names)
        if int(model.predict(df_try)[0]) == 1:
            return tweaked
        n += 3

    return fraud_dict


if __name__ == "__main__":
    from unittest.mock import patch

    agent = ThreatDetectionAgent()
    features = agent._feature_names

    safe_features: dict[str, float] = {name: 0.0 for name in features}
    try:
        # Zero vector plus strong columns from a FLAG=1 training row until RF predicts 1
        anomalous_features = _fraud_profile_for_demo(features, agent._model)
    except FileNotFoundError as e:
        print(f"[Warning] Anomaly demo: {e}")
        anomalous_features = dict(safe_features)

    vitalik = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    tether = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

    print("=== Scenario 1: Safe (clean address + zero features) ===")
    print(agent.evaluate_transaction(vitalik, safe_features))

    print("\n=== Scenario 2: Blacklisted (simulated check_blacklist=True instead of GoPlus) ===")
    fake_phishing = "0x0000000000000000000000000000000000000bad"
    with patch.object(ThreatDetectionAgent, "check_blacklist", return_value=True):
        print(agent.evaluate_transaction(fake_phishing, safe_features))

    print("\n=== Scenario 3: Anomaly (safe address + fraud-pattern features) ===")
    print(agent.evaluate_transaction(vitalik, anomalous_features))

    print(
        "\n=== Extra: Tether contract + zero features (live GoPlus; may not be blacklisted) ==="
    )
    print(agent.evaluate_transaction(tether, safe_features))
