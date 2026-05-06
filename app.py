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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from middleware import ThreatDetectionAgent
from mock_data import PROFILE_OPTIONS, PROFILE_SELECT_PLACEHOLDER, resolve_test_profile
from utils import (
    generate_network_graph,
    generate_pdf_report,
    profile_from_dataset,
    radar_dimension_names,
    radar_series,
    report_address_suffix,
    short_feature_label,
    synthetic_roc_curve_figure,
)

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "transaction_dataset.csv"
LOGO_PATH = ROOT / "logo.png"

SAFE_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

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

    Result is cached by Streamlit for responsive scenario presets.

    Args:
        flag: ``0`` for legitimate rows, ``1`` for fraud rows.
        feature_names: Expected column order from the trained model.

    Returns:
        Mapping of feature name to float value; if the CSV is missing, all zeros.
    """
    return profile_from_dataset(DATA_CSV, flag, feature_names)


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
    if "test_profile_choice" not in st.session_state:
        st.session_state.test_profile_choice = PROFILE_SELECT_PLACEHOLDER
    if "_test_profile_last_applied" not in st.session_state:
        st.session_state._test_profile_last_applied = None

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
    """
    for i, n in enumerate(names):
        st.session_state[_field_session_key(i)] = float(profile.get(n, 0.0) or 0.0)


def render_sidebar(agent: ThreatDetectionAgent) -> None:
    """Draw logo, demo controls, wallet field, and feature editors in the sidebar.

    Args:
        agent: Provides canonical feature ordering for inputs.
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

    st.markdown("### 🧪 Demo Scenarios")
    st.caption(
        "Select a preset to auto-fill **all 45 features** and the demo wallet. "
        "Phishing preset simulates blacklist mode (GoPlus call skipped)."
    )
    st.selectbox(
        "📂 Select a Test Profile",
        options=list(PROFILE_OPTIONS),
        key="test_profile_choice",
    )
    choice = st.session_state.test_profile_choice
    if choice != PROFILE_SELECT_PLACEHOLDER:
        if st.session_state.get("_test_profile_last_applied") != choice:
            addr, feat_map, force_bl = resolve_test_profile(
                choice, names, DATA_CSV, agent._model
            )
            st.session_state.ui_addr = addr
            st.session_state.demo_force_blacklist = force_bl
            push_profile_to_session(names, feat_map)
            st.session_state._test_profile_last_applied = choice
            st.session_state.pop("last_scan", None)
            st.rerun()
    else:
        st.session_state._test_profile_last_applied = None

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


def _render_ai_model_analytics_tab(agent: ThreatDetectionAgent) -> None:
    """Render tab 2: importances, synthetic ROC, and static confusion heatmap.

    Args:
        agent: Loaded agent exposing global feature importance rankings.
    """
    st.markdown("### AI Model Analytics")
    st.caption(
        "Global Random Forest structure, illustrative ROC, and offline test-set confusion for review."
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

    roc_c, cm_c = st.columns(2)
    with roc_c:
        st.subheader("ROC — True Positive vs False Positive Rate")
        st.plotly_chart(synthetic_roc_curve_figure(), use_container_width=True)
        st.caption(
            "AUC: 0.985 - Indicates excellent capability to distinguish between legitimate "
            "and fraudulent transactions."
        )
    with cm_c:
        st.subheader("Confusion Matrix (held-out test set)")
        z_cm = [[1506, 32], [4, 426]]
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
            title="Confusion Matrix (n = 1,968 test transactions)",
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
    """Render tab 3: in-memory audit trail, CSV export, and API cookbook samples.

    Uses the ``audit_logs`` list on Streamlit session state, appended after each successful scan.
    """
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
    """Render tab 1: scan workflow, metrics, XAI radar, trace graph, PDF, and raw dumps.

    Args:
        agent: Shared :class:`~middleware.ThreatDetectionAgent` (inference + GoPlus + LLM).
        names: Ordered model feature column names aligned with sidebar inputs.
    """
    with st.expander("📊 AI Model Methodology & Training Metrics", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Architecture",
                "Hybrid RF stack",
                help=(
                    "Random Forest on 45 numeric on-chain features, layered with GoPlus address "
                    "intelligence and optional Groq-generated narratives on escalation."
                ),
            )
        with m2:
            st.metric(
                "Dataset Size",
                "~19.6k samples",
                help="Labeled transaction vectors used for training and held-out validation.",
            )
        with m3:
            st.metric(
                "Overall Accuracy",
                "98.2%",
                help="Aggregate correctness on the validation split (legitimate and fraud).",
            )
        with m4:
            st.metric(
                "F1-Score",
                "0.97",
                help="Class-balanced quality metric for the fraud label (precision-recall trade-off).",
            )
        st.caption(
            "GoPlus supplies deterministic scam signals; the Random Forest scores zero-day patterns; "
            "the LLM explains outcomes for analysts when a threat path fires."
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
            dim_names = radar_dimension_names(xai_list, agent)
            cur_f = dict(snap.get("features") or {})
            baseline = _profile_from_dataset(0, tuple(names))
            r_cur, r_base = radar_series(dim_names, cur_f, baseline)
            labels = [short_feature_label(k) for k in dim_names]
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

        addr_for_graph = str(snap.get("wallet_address") or "")
        is_threat_graph = v != "SAFE"
        st.markdown("### 🕸️ Transaction Trace Graph (Local Network)")
        st.caption(
            "Illustrative *follow-the-money* style graph (synthetic layout for demos; not live chain data)."
        )
        st.plotly_chart(
            generate_network_graph(addr_for_graph, is_threat=is_threat_graph),
            use_container_width=True,
        )

        try:
            pdf_bytes = generate_pdf_report(
                str(snap.get("wallet_address") or ""),
                str(snap.get("verdict") or ""),
                float(mx_pct),
                float(snap.get("latency_ms") or 0.0),
                snap.get("xai_top_features"),
                snap.get("llm_warning"),
            )
            pdf_name = f"Threat_Report_{report_address_suffix(str(snap.get('wallet_address') or ''))}.pdf"
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
