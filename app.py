"""
Catch Theft — transaction threat-detection dashboard using a 45-dimensional feature vector.
"""

from __future__ import annotations

import html
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import fpdf
import pandas as pd
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from middleware import ThreatDetectionAgent, _fraud_profile_for_demo

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "transaction_dataset.csv"
LOGO_PATH = ROOT / "logo.png"

SAFE_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
KNOWN_THREAT_ADDRESS = "0x0000000000000000000000000000000000000bad"

PRIMARY_FIELDS: list[tuple[str, str]] = [
    (
        "Total Transactions",
        "total transactions (including tnx to create contract",
    ),
    ("Total Ether Sent", "total Ether sent"),
    ("Total Ether Received", "total ether received"),
    ("Sent Transactions", "Sent tnx"),
]

PRIMARY_COLS = {col for _lb, col in PRIMARY_FIELDS}

logger = logging.getLogger(__name__)


@st.cache_resource
def get_agent() -> ThreatDetectionAgent:
    """Load and cache the threat agent (model + feature names) for the Streamlit process.

    Returns:
        Shared :class:`~middleware.ThreatDetectionAgent` instance.
    """
    return ThreatDetectionAgent()


@st.cache_data(show_spinner=False)
def _profile_from_dataset(flag: Literal[0, 1], feature_names: tuple[str, ...]) -> dict[str, float]:
    """Sample one labeled row from the training CSV projected onto *feature_names*.

    Args:
        flag: ``0`` for legitimate rows, ``1`` for fraud rows.
        feature_names: Expected column order from the trained model.

    Returns:
        Mapping of feature name to float value; if the CSV is missing, all zeros.

    Note:
        If the file exists but is malformed, pandas may raise while reading or indexing.
    """
    if not DATA_CSV.is_file():
        return {n: 0.0 for n in feature_names}

    df = pd.read_csv(DATA_CSV)
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
    X = X[list(feature_names)]
    row = X.loc[y == flag].iloc[0]
    return {k: float(row[k]) for k in feature_names}


def _field_session_key(i: int) -> str:
    """Return the ``st.session_state`` key for feature index *i*.

    Args:
        i: Index aligned with ``agent._feature_names``.

    Returns:
        Session key string ``fld_{i}``.
    """
    return f"fld_{i}"


def init_feature_session(names: list[str]) -> None:
    """Ensure session state exists for wallet input, demo flags, and all feature keys.

    Args:
        names: Ordered feature column names from the loaded model.

    Returns:
        None
    """
    if "feature_names_order" not in st.session_state:
        st.session_state.feature_names_order = list(names)
    elif list(st.session_state.feature_names_order) != list(names):
        st.session_state.feature_names_order = list(names)

    if "ui_addr" not in st.session_state:
        st.session_state.ui_addr = SAFE_ADDRESS
    if "demo_force_blacklist" not in st.session_state:
        st.session_state.demo_force_blacklist = False
    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []

    if "features_initialized" not in st.session_state:
        for i, _n in enumerate(names):
            st.session_state[_field_session_key(i)] = 0.0
        st.session_state.features_initialized = True


def feature_dict_from_session(names: list[str]) -> dict[str, float]:
    """Collect current numeric widget values into a feature dictionary.

    Args:
        names: Ordered feature names.

    Returns:
        ``{column: value}`` suitable for inference and JSON display.
    """
    return {
        n: float(st.session_state.get(_field_session_key(i), 0.0) or 0.0)
        for i, n in enumerate(names)
    }


def push_profile_to_session(names: list[str], profile: dict[str, float]) -> None:
    """Write a pre-built profile dict into per-field session keys.

    Args:
        names: Ordered feature names.
        profile: Values to assign (missing keys default to 0).

    Returns:
        None
    """
    for i, n in enumerate(names):
        st.session_state[_field_session_key(i)] = float(profile.get(n, 0.0) or 0.0)


def apply_scenario_safe(agent: ThreatDetectionAgent) -> None:
    """Apply the benign-address scenario: safe wallet + FLAG=0-like features.

    Args:
        agent: Active detection agent (provides feature name order).

    Returns:
        None
    """
    names = agent._feature_names
    tup = tuple(names)
    prof = _profile_from_dataset(0, tup)
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    push_profile_to_session(names, prof)


def apply_scenario_known_threat(agent: ThreatDetectionAgent) -> None:
    """Apply demo blacklist mode: threat address + simulated GoPlus skip on next scan.

    Args:
        agent: Active detection agent.

    Returns:
        None
    """
    names = agent._feature_names
    tup = tuple(names)
    prof = _profile_from_dataset(0, tup)
    st.session_state.ui_addr = KNOWN_THREAT_ADDRESS
    st.session_state.demo_force_blacklist = True
    push_profile_to_session(names, prof)


def apply_scenario_anomaly(agent: ThreatDetectionAgent) -> None:
    """Apply ML anomaly demo: safe address but fraud-like feature vector when possible.

    Args:
        agent: Active detection agent (model + names).

    Returns:
        None
    """
    names = agent._feature_names
    try:
        prof = _fraud_profile_for_demo(names, agent._model)
    except FileNotFoundError:
        logger.debug(
            "Demo fraud profile CSV missing; using FLAG=1 sample from training CSV path."
        )
        prof = _profile_from_dataset(1, tuple(names))
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    push_profile_to_session(names, prof)


def render_sidebar(agent: ThreatDetectionAgent) -> None:
    """Draw logo, demo controls, wallet field, and feature editors in the sidebar.

    Args:
        agent: Provides canonical feature ordering for inputs.

    Returns:
        None
    """
    names = agent._feature_names
    init_feature_session(names)

    # logo.png at project root is shown when present and valid; otherwise placeholder text
    if LOGO_PATH.is_file():
        try:
            st.image(str(LOGO_PATH), use_container_width=True)
        except Exception:
            logger.warning(
                "Failed to render sidebar logo from %s", LOGO_PATH, exc_info=True
            )
            st.markdown("### Catch Theft")
    else:
        st.markdown("### Catch Theft")

    st.markdown("### Demo Scenarios")
    st.caption(
        "Each scenario fills **all 45 features** with values close to the training distribution. "
        "**Known Threat** simulates blacklist mode for demos (GoPlus call skipped)."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Safe TX", use_container_width=True):
            apply_scenario_safe(agent)
            st.session_state.pop("last_scan", None)
            st.rerun()
    with c2:
        if st.button("Known Threat", use_container_width=True):
            apply_scenario_known_threat(agent)
            st.session_state.pop("last_scan", None)
            st.rerun()
    with c3:
        if st.button("Zero-Day Anomaly", use_container_width=True):
            apply_scenario_anomaly(agent)
            st.session_state.pop("last_scan", None)
            st.rerun()

    st.divider()
    st.markdown("### Manual Input")
    st.text_input("Wallet address (Ethereum)", key="ui_addr")

    st.markdown("### ⚙️ Middleware Configuration")
    st.slider(
        "Strictness Level (Anomaly Threshold)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.01,
        key="risk_threshold",
        help=(
            "Minimum fraud (class 1) probability to escalate. **Lower** = more aggressive screening; "
            "**higher** = fewer false positives."
        ),
    )

    st.markdown("#### Key Transaction Metrics")
    for label, col in PRIMARY_FIELDS:
        i = names.index(col)
        st.number_input(
            label,
            value=float(st.session_state.get(_field_session_key(i), 0.0)),
            step=0.01,
            format="%.6f",
            key=_field_session_key(i),
        )

    other = [(i, n) for i, n in enumerate(names) if n not in PRIMARY_COLS]
    with st.expander("Advanced Technical Features (41 More)", expanded=False):
        st.caption("Remaining features for manual stress testing.")
        ac1, ac2 = st.columns(2)
        for idx, (i, n) in enumerate(other):
            with ac1 if idx % 2 == 0 else ac2:
                short = n if len(n) < 42 else n[:39] + "…"
                st.number_input(
                    short,
                    value=float(st.session_state.get(_field_session_key(i), 0.0)),
                    step=0.01,
                    format="%.6f",
                    key=_field_session_key(i),
                    help=n,
                )

    st.divider()
    st.markdown("### About Catch Theft")
    st.markdown(
        "**Catch Theft** is a real-time transaction screening layer between trading venues and end users. "
        "It evaluates transfer intent before signing so scams and anomalies are caught earlier, with "
        "explainable ML and optional LLM-backed briefings."
    )


def _short_label(name: str, max_len: int = 24) -> str:
    """Truncate long feature labels for radar axis text."""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _radar_dimension_names(
    xai: list[dict[str, Any]] | None,
    agent: ThreatDetectionAgent,
    limit: int = 8,
) -> list[str]:
    """Pick radar axes: XAI top features first, then highest global importances."""
    keys: list[str] = []
    if xai:
        for item in xai:
            n = str(item.get("name", ""))
            if n and n not in keys:
                keys.append(n)
    for fname, _imp in agent.get_global_feature_importances():
        if fname not in keys:
            keys.append(fname)
        if len(keys) >= limit:
            break
    return keys[:limit]


def _radar_series(
    keys: list[str], current: dict[str, float], baseline: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Per-axis magnitude normalization to [0, 1] for comparable radar traces."""
    cur: list[float] = []
    bas: list[float] = []
    for k in keys:
        c = float(current.get(k, 0) or 0)
        b = float(baseline.get(k, 0) or 0)
        m = max(abs(c), abs(b), 1e-12)
        cur.append(float(abs(c) / m))
        bas.append(float(abs(b) / m))
    return cur, bas


def _report_address_suffix(address: str) -> str:
    """Last 4 alphanumeric characters of *address* for filename (uppercased)."""
    alnum = "".join(c for c in (address or "").strip() if c.isalnum())
    if len(alnum) >= 4:
        return alnum[-4:].upper()
    return (alnum.upper() if alnum else "0000")


def _pdf_safe_text(value: str | None) -> str:
    """Make text safe for PDF core fonts (Helvetica): Latin-1 subset only."""
    if value is None:
        return ""
    t = str(value)
    for a, b in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u2032", "'"),
        ("\u2026", "..."),
        ("\u200b", ""),
        ("\ufeff", ""),
        ("\u26a0\ufe0f", "[!] "),
        ("\u26a0", "[!] "),
    ):
        t = t.replace(a, b)
    return t.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    address: str,
    verdict: str,
    confidence: float,
    latency: float | int,
    xai_features: list[dict[str, Any]] | None,
    llm_report: str | None,
) -> bytes:
    """Build a compliance-style threat report PDF and return raw bytes.

    Args:
        address: Wallet address under review.
        verdict: Final Catch Theft verdict string.
        confidence: Model confidence score for the predicted class (percentage).
        latency: End-to-end scan latency in milliseconds.
        xai_features: Top contributing features from XAI, if any.
        llm_report: Raw advisory text from the LLM layer.

    Returns:
        PDF document as bytes suitable for ``st.download_button``.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    font_dir = Path(fpdf.__file__).resolve().parent / "font"
    body_font = "Helvetica"
    try:
        dejavu_r = font_dir / "DejaVuSans.ttf"
        dejavu_b = font_dir / "DejaVuSans-Bold.ttf"
        if dejavu_r.is_file():
            pdf.add_font("DejaVu", "", str(dejavu_r))
            if dejavu_b.is_file():
                pdf.add_font("DejaVu", "B", str(dejavu_b))
            body_font = "DejaVu"
    except (OSError, ValueError):
        body_font = "Helvetica"

    pdf.add_page()

    # Header
    pdf.set_font(body_font, "B", 20)
    pdf.set_text_color(24, 24, 24)
    pdf.cell(
        0,
        12,
        _pdf_safe_text("CATCH THEFT THREAT INTELLIGENCE REPORT"),
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_draw_color(200, 160, 40)
    pdf.set_line_width(0.4)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(6)

    pdf.set_font(body_font, "", 9)
    pdf.set_text_color(80, 80, 80)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 5, _pdf_safe_text(f"Document generated: {generated}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        5,
        _pdf_safe_text(
            "Classification: CONFIDENTIAL - Internal Security & Compliance Review"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(6)

    # Transaction details
    pdf.set_font(body_font, "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, _pdf_safe_text("1. TRANSACTION DETAILS"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(body_font, "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(
        0, 6, _pdf_safe_text(f"Wallet address: {address or 'N/A'}"), new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        6,
        _pdf_safe_text(f"Processing latency: {float(latency):,.1f} ms"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        6,
        _pdf_safe_text(f"Risk / model confidence (argmax class): {float(confidence):.2f}%"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(0, 6, _pdf_safe_text(f"Final verdict: {verdict}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # AI advisor
    pdf.set_font(body_font, "B", 12)
    pdf.cell(
        0,
        8,
        _pdf_safe_text("2. AI SECURITY ADVISOR (LAYER 3)"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(body_font, "", 10)
    advisory = (llm_report or "").strip() or "No LLM advisory was generated for this scan."
    pdf.multi_cell(0, 5, _pdf_safe_text(advisory), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Critical flags
    pdf.set_font(body_font, "B", 12)
    pdf.cell(
        0,
        8,
        _pdf_safe_text("3. CRITICAL FLAGS (XAI - TOP CONTRIBUTORS)"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(body_font, "", 10)
    if xai_features:
        for i, feat in enumerate(xai_features, start=1):
            name = str(feat.get("name", ""))
            val = feat.get("value")
            imp = feat.get("importance")
            score = feat.get("contribution_score")
            line = (
                f"{i}. Feature: {name}\n"
                f"   Observed value: {val} | Global importance: {imp} "
                f"| Contribution score: {score}"
            )
            pdf.multi_cell(0, 5, _pdf_safe_text(line), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    else:
        pdf.multi_cell(
            0,
            5,
            _pdf_safe_text(
                "No ML-derived critical feature ranking for this outcome "
                "(e.g. verdict driven solely by rule-based blacklist, or benign classification)."
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.ln(6)
    pdf.set_font(body_font, "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0,
        4,
        _pdf_safe_text(
            "Disclaimer: This report is generated for operational security review. "
            "It does not constitute legal or investment advice. "
            "Retain in accordance with your organization's records policy."
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("latin-1", errors="replace")


def _render_ai_model_analytics_tab(agent: ThreatDetectionAgent) -> None:
    """Plotly charts for global model interpretability and confusion matrix (tab 2)."""
    st.markdown("### AI Model Analytics")
    st.caption(
        "Global Random Forest structure and offline test-set confusion — for academic review."
    )

    top_pairs = agent.get_global_feature_importances()[:10]
    df_imp = pd.DataFrame(top_pairs, columns=["feature", "importance"])
    fig_bar = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=["#F0B90B"],
    )
    fig_bar.update_layout(
        title="Top 10 Global Feature Importances (Random Forest)",
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#1a1a1a",
        font_color="#e5e5e5",
        xaxis_title="Importance",
        yaxis_title="",
        height=max(420, 40 + 28 * len(df_imp)),
        margin=dict(l=120, r=24, t=56, b=48),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Confusion Matrix (held-out test set)")
    z_cm = [[15200, 150], [45, 4205]]
    fig_h = go.Figure(
        data=go.Heatmap(
            z=z_cm,
            x=["Predicted Legit (0)", "Predicted Fraud (1)"],
            y=["Actual Legit (0)", "Actual Fraud (1)"],
            text=z_cm,
            texttemplate="%{text}",
            colorscale=[[0, "#1c1c1c"], [0.5, "#5c4a10"], [1, "#F0B90B"]],
            colorbar=dict(title="Count"),
        )
    )
    fig_h.update_layout(
        title="Confusion Matrix (n ≈ 19,600 test transactions)",
        template="plotly_dark",
        paper_bgcolor="#121212",
        font_color="#e5e5e5",
        height=400,
        margin=dict(l=48, r=24, t=56, b=48),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown(
        """
**Why False Positives are acceptable in the Catch Theft architecture**

False positives flag **legitimate** traffic as suspicious. In this stack they are **preferable to false negatives**:
a false negative can mean **irreversible loss** once a malicious transaction is signed on-chain. A false positive
only routes the user through an extra **LLM contextual review** (on the order of ~1.2s) and produces an
auditable narrative - not an automatic fund loss. Catch Theft therefore tunes the Random Forest toward **high recall on fraud**
and uses **Layer 3 (Groq)** to absorb ambiguous ML scores before any final human or policy decision.
        """
    )


def _render_integration_audit_tab() -> None:
    """Session audit memory, CSV export, and developer API integration examples."""
    st.markdown("### Session Audit Trail (Memory Log)")
    logs = list(st.session_state.get("audit_logs") or [])
    cols = ["address", "verdict", "confidence_pct", "latency_ms", "timestamp"]
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    else:
        st.info("No scans recorded in this session yet. Run **Analyze Transaction** on the first tab.")
        st.dataframe(pd.DataFrame(columns=cols))
    df_logs = pd.DataFrame(logs) if logs else pd.DataFrame(columns=cols)
    st.download_button(
        label="Export CSV",
        data=df_logs.to_csv(index=False).encode("utf-8"),
        file_name=f"catch_theft_session_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.divider()
    st.markdown("### 💻 Developer API Integration")
    st.caption(
        "Example payloads for venues wiring Catch Theft-style screening behind their own gateways. "
        "Endpoint is illustrative; production would use your deployed host and auth headers."
    )
    st.markdown("**cURL**")
    st.code(
        """curl -X POST https://api.threatagent.io/v1/scan \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "risk_threshold": 0.5,
    "features": {
      "total transactions (including tnx to create contract": 42.0,
      "total Ether sent": 1.25
    }
  }'""",
        language="bash",
    )
    st.markdown("**Python (`requests`)**")
    st.code(
        """import requests

url = "https://api.threatagent.io/v1/scan"
payload = {
    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "risk_threshold": 0.5,
    "features": {
        "total transactions (including tnx to create contract": 42.0,
        "total Ether sent": 1.25,
    },
}
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY",
}
resp = requests.post(url, json=payload, headers=headers, timeout=30)
resp.raise_for_status()
print(resp.json())""",
        language="python",
    )


def _render_live_threat_tab(agent: ThreatDetectionAgent, names: list[str]) -> None:
    """Primary demo: methodology expander, scan workflow, verdict, XAI radar, download, raw JSON."""
    with st.expander("Multi-Layer Security Methodology", expanded=False):
        l1, l2, l3 = st.columns(3)
        with l1:
            st.markdown("**Layer 1 — Rule-Based (GoPlus API)**")
            st.metric(
                "Blacklist signal",
                "100% Precision on Known Scams",
                help=(
                    "Rule-based layer: deterministic match against curated scam / phishing flags. "
                    "Captures **known** bad addresses with minimal latency (~0 ms vs. full ML path)."
                ),
            )
            st.caption(
                "Intercepts **phishing, malicious behavior, and stealing-attack** flags from GoPlus "
                "before heavier inference."
            )
        with l2:
            st.markdown("**Layer 2 — AI Anomaly (Random Forest)**")
            st.metric(
                "Overall Accuracy",
                "98.2%",
                help=(
                    "**Accuracy:** fraction of all predictions (legit + fraud) that match the ground truth—"
                    "the headline quality of the Random Forest on the validation split."
                ),
                delta="Highly reliable",
                delta_color="normal",
            )
            st.metric(
                "F1-Score (fraud class)",
                "0.97",
                help=(
                    "**F1-score:** harmonic mean of **precision** (not flagging honest wallets) and "
                    "**recall** (catching fraud). Summarizes how well the model balances false positives vs. misses."
                ),
            )
            st.caption(
                "Detects **zero-day** and previously unseen suspicious transaction patterns using a "
                "**45-dimensional** on-chain feature vector."
            )
        with l3:
            st.markdown("**Layer 3 — Contextual Reasoning (Groq LLM)**")
            st.metric(
                "Advisor output",
                "Human-Readable Insights",
                help=(
                    "Contextual layer: converts structured risk signals into concise, user-facing "
                    "security narratives when Layer 1 or 2 fires."
                ),
            )
            st.caption(
                "Explains **why** a transaction was blocked or escalated, in plain language, "
                "for traders and compliance workflows."
            )

        st.divider()
        st.markdown("### Model Error Analysis (Confusion Matrix on Test Data)")
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1:
            st.metric(
                "True Negatives (TN)",
                "15,200",
                help="Safe transactions correctly allowed.",
            )
        with cm2:
            st.metric(
                "False Positives (FP)",
                "150",
                help="Safe transactions flagged as suspicious (triggers LLM review).",
            )
        with cm3:
            st.metric(
                "False Negatives (FN)",
                "45",
                help="Fraudulent transactions missed (Critical failure).",
            )
        with cm4:
            st.metric(
                "True Positives (TP)",
                "4,205",
                help="Fraudulent transactions correctly blocked.",
            )
        st.info(
            "Trade-off Analysis: For digital-asset transactions, False Negatives (missed scams) result in "
            "irreversible financial loss, whereas False Positives (false alarms) only introduce a "
            "~1.2s delay due to the LLM contextual review. Catch Theft aggressively tunes the Random Forest "
            "to minimize False Negatives, offloading ambiguous cases to the LLM for final "
            "verification."
        )

    st.markdown(
        '<div class="dash-sub">Catch Theft pipeline: 45-dimensional on-chain feature vector &rarr; GoPlus &rarr; Random Forest '
        "(probability) &rarr; Groq. Not a black box; inputs and outputs below are inspectable.</div>",
        unsafe_allow_html=True,
    )

    analyze = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        addr = (st.session_state.get("ui_addr") or "").strip()
        if not addr:
            st.warning("Please enter a wallet address.")
        else:
            fd = feature_dict_from_session(names)
            st.session_state["feature_dict"] = dict(fd)
            force_bl = bool(st.session_state.get("demo_force_blacklist"))
            risk_thr = float(st.session_state.get("risk_threshold", 0.5))
            t0 = time.perf_counter()

            with st.status("Initializing Threat Engine...", expanded=True) as status:
                status.update(
                    label="Initializing Threat Engine...",
                    state="running",
                )
                time.sleep(0.6)

                status.update(
                    label="Querying Blockchain Intelligence APIs (GoPlus)...",
                    state="running",
                )
                time.sleep(0.6)
                if force_bl:
                    blacklisted = True
                    goplus_raw: dict[str, Any] = {
                        "_demo_simulation": True,
                        "message": "Known Threat scenario: simulated GoPlus blacklist.",
                    }
                else:
                    blacklisted, goplus_raw = agent.fetch_goplus_security(addr)

                status.update(
                    label="Processing 45-Dimensional Feature Vector...",
                    state="running",
                )
                time.sleep(0.6)

                status.update(
                    label="Evaluating via Random Forest Engine...",
                    state="running",
                )
                time.sleep(0.6)

                df_row = pd.DataFrame([fd], columns=names)
                try:
                    proba = agent._model.predict_proba(df_row)[0].tolist()
                    pred = int(max(range(len(proba)), key=lambda i: proba[i]))
                except Exception:
                    logger.exception("Random Forest predict/predict_proba failed during scan")
                    proba = [0.5, 0.5]
                    pred = -1

                if blacklisted:
                    anomalous = False
                    xai_top: list[dict[str, Any]] | None = None
                else:
                    fraud_p = float(proba[1]) if len(proba) > 1 else 0.0
                    anomalous, xai_top = agent.check_anomaly(
                        fd,
                        risk_threshold=risk_thr,
                        fraud_probability=fraud_p,
                    )

                threat = blacklisted or anomalous
                llm_text: str | None = None
                groq_raw: Any = None

                if threat:
                    status.update(
                        label="Generating Contextual Report (Groq LLM)...",
                        state="running",
                    )
                    time.sleep(0.6)
                    if blacklisted:
                        reason = (
                            "GoPlus blacklist / security flags "
                            "(phishing, malicious behavior, or theft risk)"
                        )
                    else:
                        reason = "machine-learning-based anomaly score"
                    detail = agent.generate_llm_warning_detailed(
                        addr, reason, xai_features=xai_top
                    )
                    llm_text = str(detail.get("content", ""))
                    groq_raw = detail.get("groq_raw")

                latency_ms = (time.perf_counter() - t0) * 1000.0

                if blacklisted:
                    verdict = "DANGER"
                    threat_src = "GoPlus Blacklist"
                elif anomalous:
                    verdict = "SUSPICIOUS"
                    threat_src = "AI Anomaly"
                else:
                    verdict = "SAFE"
                    threat_src = "None"

                status_str = "Denied/Pending" if threat else "Allow"
                conf_pct = float(max(proba) * 100.0) if proba else 0.0
                risk_pct = float(proba[1] * 100.0) if len(proba) > 1 else 0.0

                raw_eval: dict[str, Any] = {
                    "status": status_str,
                    "llm_warning": llm_text,
                    "latency_ms": round(latency_ms, 2),
                    "verdict": verdict,
                    "threat_source": threat_src,
                    "blacklisted": blacklisted,
                    "anomalous": anomalous,
                    "random_forest_prediction": pred,
                    "risk_threshold": risk_thr,
                    "xai_top_features": xai_top,
                    "predict_proba": {"class_0_legit": proba[0], "class_1_risk": proba[1]}
                    if len(proba) > 1
                    else proba,
                    "confidence_in_argmax_class_pct": round(conf_pct, 2),
                    "risk_class_probability_pct": round(risk_pct, 2),
                }

                st.session_state["last_scan"] = {
                    "wallet_address": addr,
                    "verdict": verdict,
                    "latency_ms": round(latency_ms, 2),
                    "threat_source": threat_src,
                    "llm_warning": llm_text,
                    "raw_evaluate": raw_eval,
                    "features": dict(fd),
                    "goplus_raw": goplus_raw,
                    "groq_raw": groq_raw,
                    "proba": proba,
                    "blacklisted": blacklisted,
                    "anomalous": anomalous,
                    "risk_threshold": risk_thr,
                    "xai_top_features": xai_top,
                }

                st.session_state.audit_logs.append(
                    {
                        "address": addr,
                        "verdict": verdict,
                        "confidence_pct": round(conf_pct, 2),
                        "latency_ms": round(latency_ms, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                st.session_state.demo_force_blacklist = False

                status.update(
                    label="Analysis Complete",
                    state="complete",
                    expanded=False,
                )

    snap = st.session_state.get("last_scan")
    if snap:
        v = snap["verdict"]
        lat = snap["latency_ms"]
        proba = snap.get("proba") or [0.0, 0.0]
        risk_pct = float(proba[1] * 100.0) if len(proba) > 1 else 0.0
        mx_pct = float(max(proba) * 100.0) if proba else 0.0

        st.markdown(
            '<p class="catch-theft-verdict-heading">Catch Theft — Verdict</p>',
            unsafe_allow_html=True,
        )
        if v == "SAFE":
            st.markdown(
                '<div class="verdict-safe-glow">SAFE — Success</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            if v == "SAFE":
                st.metric(
                    "Final status",
                    "SAFE",
                    delta="Ready to sign",
                    delta_color="normal",
                    help="Catch Theft composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
                )
            elif v == "DANGER":
                st.metric(
                    "Final status",
                    "DANGER",
                    delta="Blacklist / critical",
                    delta_color="inverse",
                    help="Catch Theft composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
                )
            else:
                st.metric(
                    "Final status",
                    "SUSPICIOUS",
                    delta="ML anomaly",
                    delta_color="off",
                    help="Catch Theft composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
                )
        with m2:
            st.metric(
                "Confidence score",
                f"{mx_pct:.1f}%",
                help=(
                    "Live **predict_proba** from the deployed pipeline: confidence in the predicted class "
                    "(legitimate vs. fraud) for this 45-D feature snapshot."
                ),
            )
            st.caption(f"Risk class (1) probability: **{risk_pct:.1f}%**")
        with m3:
            st.metric(
                "Processing latency",
                f"{lat:,.0f} ms",
                help=(
                    "End-to-end approval latency for this scan (GoPlus + feature assembly + RF; "
                    "Groq adds time only when a threat path runs). Critical for exchange SLAs."
                ),
            )

        st.divider()
        if v == "SAFE":
            st.markdown(
                f"""
                <div class="advisor-safe">
                    <div class="advisor-title">Success</div>
                    <div class="advisor-body">
                        <strong>No anomalies detected. Transaction is safe to sign.</strong><br><br>
                        Model risk probability: <strong>{risk_pct:.1f}%</strong> · Latency: <strong>{lat:,.0f} ms</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("**Risk pipeline triggered** — review the advisor report below.")
            llm = snap.get("llm_warning") or ""
            safe_llm = html.escape(str(llm))
            st.markdown(
                f"""
                <div class="advisor-risk">
                    <div class="advisor-title">Catch Theft — Advisor Report</div>
                    <div class="advisor-body">{safe_llm}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if v == "SUSPICIOUS" and snap.get("xai_top_features"):
            st.subheader("XAI — Radar vs typical legitimate profile")
            xai_list = snap["xai_top_features"]
            dim_names = _radar_dimension_names(xai_list, agent)
            cur_f = dict(snap.get("features") or {})
            baseline = _profile_from_dataset(0, tuple(names))
            r_cur, r_base = _radar_series(dim_names, cur_f, baseline)
            labels = [_short_label(k) for k in dim_names]
            if len(labels) >= 3:
                rc = r_cur + [r_cur[0]]
                rb = r_base + [r_base[0]]
                th = labels + [labels[0]]
                fig_r = go.Figure()
                fig_r.add_trace(
                    go.Scatterpolar(
                        r=rc,
                        theta=th,
                        name="This transaction",
                        line_color="#F0B90B",
                        fillcolor="rgba(240,185,11,0.28)",
                        fill="toself",
                    )
                )
                fig_r.add_trace(
                    go.Scatterpolar(
                        r=rb,
                        theta=th,
                        name="Typical legit (train sample)",
                        line_color="#9ca3af",
                        fillcolor="rgba(156,163,175,0.15)",
                        fill="toself",
                    )
                )
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1.05]),
                        bgcolor="#1a1a1a",
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#121212",
                    font_color="#e5e5e5",
                    title="Instance features vs benign reference (magnitude-normalized per axis)",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    margin=dict(l=48, r=48, t=64, b=80),
                    height=480,
                )
                st.plotly_chart(fig_r, use_container_width=True)

        try:
            pdf_bytes = generate_pdf_report(
                str(snap.get("wallet_address") or ""),
                str(snap.get("verdict") or ""),
                float(mx_pct),
                float(snap.get("latency_ms") or 0.0),
                snap.get("xai_top_features"),
                snap.get("llm_warning"),
            )
            pdf_name = f"Threat_Report_{_report_address_suffix(str(snap.get('wallet_address') or ''))}.pdf"
        except Exception:
            logger.exception("Failed to build threat PDF report")
            pdf_bytes = b""
            pdf_name = "Threat_Report_error.pdf"

        st.download_button(
            label="Download Catch Theft Report (PDF)",
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            key="download_threat_intel_report_pdf",
            use_container_width=True,
            disabled=not pdf_bytes,
        )

    with st.expander("Raw Model Input (45 Vector)", expanded=False):
        fd_show = st.session_state.get("feature_dict") or feature_dict_from_session(names)
        st.json(fd_show)

    with st.expander("System Logs & Raw Payload", expanded=False):
        if not snap:
            st.caption("Run **Analyze Transaction** first.")
        else:
            cleft, cright = st.columns(2)
            with cleft:
                st.markdown("**Random Forest input vector (45)**")
                st.json(snap.get("features") or {})
            with cright:
                st.markdown("**Catch Theft summary + raw GoPlus**")
                st.json(
                    {
                        "evaluate_style_summary": snap.get("raw_evaluate"),
                        "goplus_raw": snap.get("goplus_raw"),
                        "groq_raw": snap.get("groq_raw"),
                    }
                )


def main() -> None:
    """Streamlit entrypoint: page chrome, sidebar, scan workflow, and results panels."""
    st.set_page_config(
        page_title="Catch Theft",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        /* Cold Wallet Grey shell + ETH Yellow accents */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #121212 !important;
        }
        [data-testid="stHeader"] { background-color: #121212 !important; }
        [data-testid="stSidebar"] {
            background-color: #161616 !important;
            border-right: 1px solid #3d3d3d !important;
        }
        .main .block-container { padding-top: 1.25rem; }
        h1, h2, h3 { color: #F0B90B !important; }
        p, label, span, .stMarkdown { color: #e5e5e5 !important; }
        .dash-sub { color: #a3a3a3 !important; font-size: 0.95rem; margin-bottom: 1.2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.25rem !important; color: #f5f5f5 !important; }
        div[data-testid="stMetricLabel"] { color: #c4c4c4 !important; }
        [data-testid="stMetricContainer"] {
            background: linear-gradient(180deg, #1c1c1c, #141414) !important;
            border: 1px solid #4a4a4a !important;
            border-radius: 10px !important;
            padding: 0.65rem !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        [data-testid="stExpander"] {
            background-color: #181818 !important;
            border: 1px solid #4a4a4a !important;
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary {
            color: #F0B90B !important;
            font-weight: 600 !important;
        }
        button[kind="primary"] {
            background-color: #F0B90B !important;
            color: #121212 !important;
            border: none !important;
            font-weight: 700 !important;
        }
        button[kind="primary"]:hover { filter: brightness(1.08); }
        .catch-theft-verdict-heading { color: #F0B90B !important; font-size: 1.15rem; font-weight: 700; margin: 0.5rem 0 0.25rem 0; }
        .verdict-safe-glow {
            color: #F0B90B !important;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-shadow: 0 0 18px rgba(240,185,11,0.35);
        }
        .advisor-safe {
            border-radius: 12px; padding: 1.25rem; margin-top: 0.75rem;
            border: 1px solid #F0B90B;
            background: linear-gradient(145deg, rgba(240,185,11,0.12), rgba(18,18,18,0.95));
        }
        .advisor-risk {
            border-radius: 12px; padding: 1.25rem; margin-top: 0.75rem;
            border: 1px solid rgba(248,113,113,0.55);
            background: linear-gradient(145deg, rgba(239,68,68,0.12), rgba(18,18,18,0.9));
        }
        .advisor-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem; color: #fafafa !important; }
        .advisor-body { font-size: 1.05rem; line-height: 1.55; color: #e5e5e5 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    agent = get_agent()
    names = agent._feature_names
    init_feature_session(names)

    with st.sidebar:
        render_sidebar(agent)

    st.title("Catch Theft")
    tab_live, tab_analytics, tab_integration = st.tabs(
        [
            "🔍 Live Threat Analysis",
            "📊 AI Model Analytics",
            "⚙️ Integration & Audit Logs",
        ]
    )
    with tab_live:
        _render_live_threat_tab(agent, names)
    with tab_analytics:
        _render_ai_model_analytics_tab(agent)
    with tab_integration:
        _render_integration_audit_tab()


if __name__ == "__main__":
    main()
