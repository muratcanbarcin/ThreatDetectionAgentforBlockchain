"""
Hybrid Detection Engine: GoPlus kara liste + Random Forest anomali + Groq uyarısı.
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
        """GoPlus API tam JSON yanıtı ve kara liste kararı (UI / log için)."""
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
            f"Şu cüzdan adresi ({address}) {threat_reason} nedeniyle riskli bulundu. "
            "Durumu değrlendir ve eğer riskli ise işlemi iptal etmesini söyleyen 2-3 cümlelik profesyonel bir "
            "siber güvenlik uyarısı yaz veya eğer riskli değilse işlemi onayla"
            "Her bir sonucun için bir güven skoru ver."
            "Güven skoru 0-100 arasında bir sayıdır."
            "0-20: Çok düşük güven"
            "21-40: Düşük güven"
            "41-60: Orta güven"
            "61-80: Yüksek güven"
            "81-100: Çok yüksek güven"
            "Güven skoru 80 ve üzeri ise işlemi onayla, 80'in altında ise işlemi iptal etmeyi öner"
        )

        if self._groq is None:
            text = (
                "Uyarı: GROQ_API_KEY tanımlı değil; LLM devre dışı. "
                f"Adres {address} için {threat_reason} tespit edildi; işlemi iptal etmeniz önerilir."
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
                "Uyarı: Groq çağrısı başarısız (ağ veya kota). "
                f"Adres {address} için {threat_reason} tespit edildi; işlemi iptal etmeniz önerilir."
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
                reason = "GoPlus kara liste / güvenlik bayrakları (phishing, kötü amaçlı davranış veya hırsızlık riski)"
            else:
                reason = "makine öğrenmesi tabanlı anomali skoru"

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
    """Sıfır tabanlı mock üzerine birkaç sütunu FLAG=1 örneğinden doldurur; RF tahmini 1 olana kadar genişletir."""
    csv_path = ROOT / "data" / "transaction_dataset.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Demo için veri seti bulunamadı: {csv_path}")

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
        # Sıfır vektör + eğitimdeki FLAG=1 profilinden seçilen birkaç güçlü sütun (RF=1 için)
        anomalous_features = _fraud_profile_for_demo(features, agent._model)
    except FileNotFoundError as e:
        print(f"[Uyarı] Anomali demo: {e}")
        anomalous_features = dict(safe_features)

    vitalik = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    tether = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

    print("=== Senaryo 1: Safe (temiz adres + sıfır özellikler) ===")
    print(agent.evaluate_transaction(vitalik, safe_features))

    print("\n=== Senaryo 2: Blacklisted (GoPlus yerine check_blacklist=True simülasyonu) ===")
    fake_phishing = "0x0000000000000000000000000000000000000bad"
    with patch.object(ThreatDetectionAgent, "check_blacklist", return_value=True):
        print(agent.evaluate_transaction(fake_phishing, safe_features))

    print("\n=== Senaryo 3: Anomaly (güvenli adres + dolandırıcılık örüntülü özellikler) ===")
    print(agent.evaluate_transaction(vitalik, anomalous_features))

    print("\n=== Ek: Tether kontratı + sıfır özellik (GoPlus canlı yanıt, kara liste olmayabilir) ===")
    print(agent.evaluate_transaction(tether, safe_features))
