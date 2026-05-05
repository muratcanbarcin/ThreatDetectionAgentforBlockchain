"""
Catch Theft — hybrid detection: GoPlus rule layer + Random Forest anomaly scoring + Groq advisory.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"
FEATURES_PATH = MODELS_DIR / "model_features.pkl"
GOPLUS_TEMPLATE = (
    "https://api.gopluslabs.io/api/v1/address_security/{address}?chain_id=1"
)
GOPLUS_TIMEOUT_SEC = 15

logger = logging.getLogger(__name__)

LLM_FALLBACK_TIMEOUT = (
    "⚠️ LLM Contextual Analysis is currently unavailable due to network timeout."
)
LLM_FALLBACK_GENERIC = (
    "⚠️ LLM Contextual Analysis is currently unavailable due to a network or "
    "service error."
)


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True if *exc* represents a timeout-style failure from HTTP or SDK clients."""
    if isinstance(exc, TimeoutError):
        return True
    et = type(exc).__name__.lower()
    if "timeout" in et:
        return True
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg


def _classifier_from_pipeline(model: Any) -> Any:
    """Return the final estimator (RandomForest) from a sklearn Pipeline or the model itself."""
    if hasattr(model, "named_steps"):
        steps = getattr(model, "named_steps", {})
        if isinstance(steps, dict) and "classifier" in steps:
            return steps["classifier"]
        if isinstance(steps, dict) and len(steps) > 0:
            return steps[list(steps.keys())[-1]]
    return model


class ThreatDetectionAgent:
    """Catch Theft core agent: GoPlus lookups, Random Forest scoring, and optional Groq text generation."""

    def __init__(self) -> None:
        """Load the serialized Random Forest pipeline and feature name order from disk.

        Raises:
            FileNotFoundError: If ``models/rf_model.pkl`` or ``models/model_features.pkl`` is missing.
            OSError: If model files cannot be read.
        """
        self._model = joblib.load(MODEL_PATH)
        self._feature_names: list[str] = list(joblib.load(FEATURES_PATH))

        api_key = os.environ.get("GROQ_API_KEY")
        self._groq: Groq | None = Groq(api_key=api_key) if api_key else None

    def _rf_estimator(self) -> Any:
        """Resolved RandomForest (or final step) used for ``feature_importances_``."""
        return _classifier_from_pipeline(self._model)

    def _feature_importance_array(self) -> list[float]:
        """Global feature importances aligned with :attr:`_feature_names`."""
        clf = self._rf_estimator()
        imps = getattr(clf, "feature_importances_", None)
        if imps is None:
            return [1.0 / max(1, len(self._feature_names))] * len(self._feature_names)
        arr = cast(Any, imps)
        return [float(x) for x in arr.tolist()]

    def get_global_feature_importances(self) -> list[tuple[str, float]]:
        """Feature names with global RF importances, sorted descending.

        Returns:
            Ordered list of ``(feature_name, importance)`` pairs.
        """
        pairs = list(zip(self._feature_names, self._feature_importance_array()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def _top_critical_features(
        self, features_dict: dict[str, Any], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Rank features by ``importance × |value|`` (instance-level XAI heuristic).

        Args:
            features_dict: Current transaction features.
            top_k: Number of leading contributors to return.

        Returns:
            List of dicts with ``name``, ``value``, ``importance``, ``contribution_score``.
        """
        imps = self._feature_importance_array()
        scored: list[tuple[str, float, float, float]] = []
        for name, imp in zip(self._feature_names, imps):
            val = float(features_dict.get(name, 0) or 0)
            contribution = float(imp) * abs(val)
            scored.append((name, val, float(imp), contribution))
        scored.sort(key=lambda x: x[3], reverse=True)
        if scored and scored[0][3] == 0.0:
            scored.sort(key=lambda x: x[2], reverse=True)
        out = []
        for name, val, imp, contrib in scored[:top_k]:
            out.append(
                {
                    "name": name,
                    "value": val,
                    "importance": imp,
                    "contribution_score": contrib,
                }
            )
        return out

    def top_critical_features(
        self, features_dict: dict[str, Any], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Public alias for top contributing features (use when fraud class is already predicted).

        Args:
            features_dict: Feature map for the transaction under review.
            top_k: How many features to return.

        Returns:
            Top-*k* features by contribution heuristic.
        """
        return self._top_critical_features(features_dict, top_k=top_k)

    def fetch_goplus_security(self, address: str) -> tuple[bool, dict[str, Any]]:
        """Query GoPlus address security and derive a blacklist boolean from flag fields.

        Args:
            address: Ethereum ``0x`` address (mainnet in template URL).

        Returns:
            Tuple of ``(blacklisted, payload_or_error_dict)``. On transport/HTTP errors,
            ``blacklisted`` is ``False`` and the second element contains ``error`` and ``source`` keys.

        Raises:
            Nothing: Errors are swallowed and returned as structured dicts for UI stability.
        """
        url = GOPLUS_TEMPLATE.format(address=address)
        try:
            resp = requests.get(url, timeout=GOPLUS_TIMEOUT_SEC)
            if resp.status_code >= 500:
                logger.error(
                    "GoPlus server error %s for address prefix=%s",
                    resp.status_code,
                    (address[:18] + "…") if len(address) > 18 else address,
                )
                return False, {
                    "error": f"HTTP {resp.status_code} from GoPlus",
                    "source": "goplus_http",
                }
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        except requests.Timeout:
            logger.warning(
                "GoPlus request timed out after %ss (address prefix=%s)",
                GOPLUS_TIMEOUT_SEC,
                (address[:18] + "…") if len(address) > 18 else address,
            )
            return False, {"error": "Request timed out", "source": "goplus_http"}
        except requests.HTTPError as e:
            logger.warning("GoPlus HTTP error: %s", e)
            return False, {"error": str(e), "source": "goplus_http"}
        except requests.RequestException as e:
            logger.warning("GoPlus request failed: %s", e)
            return False, {"error": str(e), "source": "goplus_http"}
        except ValueError as e:
            logger.warning("GoPlus JSON decode error: %s", e)
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
        """Return True if GoPlus reports phishing/malicious/stealing flags for this address.

        Args:
            address: Wallet address to check.

        Returns:
            ``True`` if any relevant GoPlus flag is set; ``False`` on safe result or any failure
            (fail-open to avoid blocking traffic when the API is down).

        Raises:
            Nothing.
        """
        try:
            threat, _payload = self.fetch_goplus_security(address)
            return threat
        except Exception:
            logger.exception(
                "Unexpected error in check_blacklist for address prefix=%s",
                (address[:18] + "…") if len(address) > 18 else address,
            )
            return False

    def check_anomaly(
        self,
        features_dict: dict[str, Any],
        risk_threshold: float = 0.5,
        *,
        fraud_probability: float | None = None,
    ) -> tuple[bool, list[dict[str, Any]] | None]:
        """Flag anomaly when fraud (class 1) probability meets or exceeds *risk_threshold*.

        Uses ``predict_proba`` (not hard ``predict``). When *fraud_probability* is provided
        (e.g. from a shared pipeline call in the UI), the model is not queried again.

        Args:
            features_dict: Mapping from feature column name to numeric value.
            risk_threshold: Minimum P(fraud) to treat as anomalous (0.0-1.0).
            fraud_probability: Optional precomputed P(class=1); if ``None``, computed here.

        Returns:
            ``(is_anomaly, top_features)`` where ``top_features`` is ``None`` if below threshold.

        Raises:
            ValueError: Propagated by pandas/sklearn if the feature set is invalid (unexpected in normal UI use).
        """
        thr = float(risk_threshold)
        if fraud_probability is not None:
            p1 = float(fraud_probability)
        else:
            row = {
                name: float(features_dict.get(name, 0) or 0)
                for name in self._feature_names
            }
            df = pd.DataFrame([row], columns=self._feature_names)
            proba = self._model.predict_proba(df)[0]
            p1 = float(proba[1]) if len(proba) > 1 else 0.0
        if p1 < thr:
            return False, None
        return True, self._top_critical_features(features_dict, top_k=3)

    def generate_llm_warning_detailed(
        self,
        address: str,
        threat_reason: str,
        xai_features: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a Groq chat completion with a structured security prompt.

        Args:
            address: Wallet address shown in the advisory.
            threat_reason: Human-readable reason from Layer 1 or 2.
            xai_features: Optional top Random Forest features (name, value, importance) for data-driven text.

        Returns:
            Dict with keys ``content`` (advisory text), ``groq_raw`` (API payload or error stub),
            and ``used_llm`` (bool).

        Raises:
            Nothing: Failures produce user-visible fallback strings instead of raising.
        """
        xai_block = ""
        if xai_features:
            lines = []
            for item in xai_features:
                lines.append(
                    f"- Feature '{item['name']}': observed value={item['value']:.6g}, "
                    f"global model importance={float(item['importance']):.6g}, "
                    f"instance contribution score (importance×|value|)={float(item['contribution_score']):.6g}"
                )
            xai_block = (
                "\n\nData-driven context from the Random Forest (instance-level XAI - top contributors):\n"
                + "\n".join(lines)
                + "\n\nGround your explanation in these concrete numbers: compare them to what you would "
                "expect for a typical legitimate on-chain wallet (e.g. unusually high send volume vs. "
                "account/history signals). Do not use generic boilerplate; cite at least one feature by name."
            )

        prompt = (
            f"This wallet address ({address}) was flagged as risky because: {threat_reason}.{xai_block}\n\n"
            "Write a concise professional cybersecurity advisory (2–4 sentences) that is explicitly "
            "informed by the data above when XAI context is present. "
            "Tell the user whether to cancel the transaction if risk remains credible, or approve if not. "
            "End with a confidence score 0–100 (single line: 'Confidence: XX/100') using: "
            "0-20 very low; 21-40 low; 41-60 medium; 61-80 high; 81-100 very high. "
            "If confidence is 80+ suggest approval only if justified; below 80 recommend cancellation or manual review."
        )

        if self._groq is None:
            text = (
                "Warning: GROQ_API_KEY is not set; LLM is disabled. "
                f"For address {address}, {threat_reason} was detected; canceling the transaction is recommended."
            )
            logger.info("Groq client unavailable (no API key); returning static advisory.")
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
            if _is_timeout_error(e):
                logger.warning("Groq request timed out: %s", e)
                text = LLM_FALLBACK_TIMEOUT
            else:
                logger.error("Groq request failed: %s", e, exc_info=True)
                text = LLM_FALLBACK_GENERIC
            return {"content": text, "groq_raw": {"error": str(e)}, "used_llm": False}

    def generate_llm_warning(
        self,
        address: str,
        threat_reason: str,
        xai_features: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return only the advisory text from :meth:`generate_llm_warning_detailed`.

        Args:
            address: Wallet address.
            threat_reason: Reason string for the model prompt.
            xai_features: Optional XAI feature list passed to the LLM.

        Returns:
            Non-empty advisory or fallback message if the detailed path fails unexpectedly.

        Raises:
            Nothing.
        """
        try:
            return str(
                self.generate_llm_warning_detailed(
                    address, threat_reason, xai_features=xai_features
                )["content"]
            )
        except Exception:
            logger.exception("generate_llm_warning failed for address prefix=%s", address[:18])
            return LLM_FALLBACK_GENERIC

    def evaluate_transaction(
        self,
        address: str,
        features_dict: dict[str, Any],
        risk_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """End-to-end single-transaction evaluation (GoPlus + RF + conditional LLM).

        Args:
            address: On-chain address under review.
            features_dict: Numeric features aligned with training column order.
            risk_threshold: Minimum P(fraud) from :meth:`check_anomaly` to treat as denied.

        Returns:
            Dict with ``status`` (``\"Allow\"`` or ``\"Denied/Pending\"``), optional ``llm_warning``,
            ``latency_ms``, and ``xai_top_features`` when ML anomaly fires.

        Raises:
            Nothing under normal API/ML failure modes; ML errors could propagate from :meth:`check_anomaly`.
        """
        t0 = time.perf_counter()

        blacklisted = self.check_blacklist(address)
        if blacklisted:
            anomalous, xai_top = False, None
        else:
            anomalous, xai_top = self.check_anomaly(
                features_dict, risk_threshold=risk_threshold
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if blacklisted or anomalous:
            if blacklisted:
                reason = (
                    "GoPlus blacklist / security flags "
                    "(phishing, malicious behavior, or theft risk)"
                )
                xai_for_llm: list[dict[str, Any]] | None = None
            else:
                reason = "machine-learning-based anomaly score"
                xai_for_llm = xai_top

            llm_text = self.generate_llm_warning(address, reason, xai_features=xai_for_llm)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "status": "Denied/Pending",
                "llm_warning": llm_text,
                "latency_ms": round(latency_ms, 2),
                "xai_top_features": xai_top,
            }

        return {
            "status": "Allow",
            "llm_warning": None,
            "latency_ms": round(latency_ms, 2),
            "xai_top_features": None,
        }


def _fraud_profile_for_demo(
    feature_names: list[str], model: Any, max_tweaks: int = 12
) -> dict[str, float]:
    """Construct a feature vector that triggers fraud prediction for UI demos.

    Starting from zeros, progressively copies the strongest values from one known
    fraud row until ``model`` predicts class 1.

    Args:
        feature_names: Ordered list of column names expected by the trained model.
        model: Fitted sklearn-compatible estimator with ``predict``.
        max_tweaks: Initial number of features to copy from the fraud profile.

    Returns:
        Feature dict that yields a positive (fraud) prediction, or the full fraud profile.

    Raises:
        FileNotFoundError: If ``data/transaction_dataset.csv`` does not exist.
        KeyError: If required columns are missing from the dataset.
    """
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from unittest.mock import patch

    agent = ThreatDetectionAgent()
    features = agent._feature_names

    safe_features: dict[str, float] = {name: 0.0 for name in features}
    try:
        anomalous_features = _fraud_profile_for_demo(features, agent._model)
    except FileNotFoundError as e:
        logger.warning("Anomaly demo profile unavailable: %s", e)
        anomalous_features = dict(safe_features)

    vitalik = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    tether = "0xdAC17F958D2ee523a2206206994597C13D831ec7"

    logger.info("=== Scenario 1: Safe (clean address + zero features) ===")
    logger.info("%s", agent.evaluate_transaction(vitalik, safe_features))

    logger.info("=== Scenario 2: Blacklisted (simulated check_blacklist=True) ===")
    fake_phishing = "0x0000000000000000000000000000000000000bad"
    with patch.object(ThreatDetectionAgent, "check_blacklist", return_value=True):
        logger.info("%s", agent.evaluate_transaction(fake_phishing, safe_features))

    logger.info("=== Scenario 3: Anomaly (safe address + fraud-pattern features) ===")
    logger.info("%s", agent.evaluate_transaction(vitalik, anomalous_features))

    logger.info(
        "=== Extra: Tether contract + zero features (live GoPlus; may not be blacklisted) ==="
    )
    logger.info("%s", agent.evaluate_transaction(tether, safe_features))
