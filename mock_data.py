"""
Preset test profiles for the Catch Theft UI (CSV-backed and synthetic blends).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from middleware import _fraud_profile_for_demo
from utils import profile_from_dataset

PROFILE_SELECT_PLACEHOLDER: Final[str] = "Select a profile..."

PROFILE_OPTIONS: Final[tuple[str, ...]] = (
    PROFILE_SELECT_PLACEHOLDER,
    "1. Normal User (Low Volume)",
    "2. Normal User (Active Trader)",
    "3. Known Phishing Address",
    "4. Zero-Day Bot Anomaly",
    "5. Dormant Account Abuse",
)

SAFE_DEMO_WALLET: Final[str] = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
PHISHING_DEMO_WALLET: Final[str] = "0x0000000000000000000000000000000000000bad"


def resolve_test_profile(
    choice: str,
    feature_names: list[str],
    csv_path: Path,
    model: Any,
) -> tuple[str, dict[str, float], bool]:
    """Build wallet address, 45-D feature dict, and optional demo blacklist flag for a preset.

    Args:
        choice: One of :data:`PROFILE_OPTIONS` (excluding the placeholder).
        feature_names: Model column order.
        csv_path: Training CSV path for :func:`utils.profile_from_dataset`.
        model: Fitted estimator for :func:`middleware._fraud_profile_for_demo`.

    Returns:
        ``(ethereum_address, features, demo_force_blacklist)``.

    Raises:
        ValueError: If *choice* is unknown.
        FileNotFoundError: Propagated when the fraud demo builder cannot access the CSV (handled upstream in callers).
    """
    tup = tuple(feature_names)
    if choice == "1. Normal User (Low Volume)":
        base = profile_from_dataset(csv_path, 0, tup)
        feat = {k: float(v) * 0.12 for k, v in base.items()}
        return SAFE_DEMO_WALLET, feat, False

    if choice == "2. Normal User (Active Trader)":
        base = profile_from_dataset(csv_path, 0, tup)
        feat = {}
        for k, v in base.items():
            kl = k.lower()
            if any(
                s in kl
                for s in (
                    "transaction",
                    "ether",
                    "tnx",
                    "sent",
                    "received",
                    "contract",
                )
            ):
                feat[k] = float(v) * 2.3 + (0.25 if float(v) < 1e-6 else 0.0)
            else:
                feat[k] = float(v) * 1.15
        return SAFE_DEMO_WALLET, feat, False

    if choice == "3. Known Phishing Address":
        base = profile_from_dataset(csv_path, 0, tup)
        return PHISHING_DEMO_WALLET, base, True

    if choice == "4. Zero-Day Bot Anomaly":
        try:
            feat = _fraud_profile_for_demo(feature_names, model)
        except FileNotFoundError:
            feat = profile_from_dataset(csv_path, 1, tup)
        return SAFE_DEMO_WALLET, feat, False

    if choice == "5. Dormant Account Abuse":
        legit = profile_from_dataset(csv_path, 0, tup)
        fraud = profile_from_dataset(csv_path, 1, tup)
        feat = {k: float(legit[k]) * 0.04 for k in feature_names}
        for k in feature_names:
            kl = k.lower()
            if any(s in kl for s in ("sent", "transaction", "tnx")):
                feat[k] = max(feat[k], float(legit[k]) * 6.0 + 12.0)
        for k in feature_names:
            feat[k] = 0.72 * feat[k] + 0.28 * float(fraud[k])
        return SAFE_DEMO_WALLET, feat, False

    raise ValueError(f"Unknown test profile: {choice!r}")
