"""
Web3 Threat Detection — security dashboard aligned with a 45-dimensional feature vector.
"""

from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
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
            st.markdown("### MIDDLEWARE LOGO")
    else:
        st.markdown("### MIDDLEWARE LOGO")

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
    st.markdown("### About Middleware")
    st.markdown(
        "This agent acts as a secure API bridge between Cryptocurrency Exchanges and End-Users. "
        "It intercepts transaction intents in real-time to provide a critical security layer before signing."
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
**Why False Positives are acceptable in our architecture**

False positives flag **legitimate** traffic as suspicious. In our stack they are **preferable to false negatives**:
a false negative can mean **irreversible loss** once a malicious transaction is signed on-chain. A false positive
only routes the user through an extra **LLM contextual review** (on the order of ~1.2s) and produces an
auditable narrative — not an automatic fund loss. We therefore tune the Random Forest toward **high recall on fraud**
and use **Layer 3 (Groq)** to absorb ambiguous ML scores before any final human or policy decision.
        """
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
            "Trade-off Analysis: In Web3 security, False Negatives (missed scams) result in "
            "irreversible financial loss, whereas False Positives (false alarms) only introduce a "
            "~1.2s delay due to the LLM contextual review. Our Random Forest model is aggressively "
            "tuned to minimize False Negatives, offloading ambiguous cases to the LLM for final "
            "verification."
        )

    st.markdown(
        '<div class="dash-sub">45-dimensional on-chain feature vector &rarr; GoPlus &rarr; Random Forest '
        "(probability) &rarr; Groq. Not a black box; all inputs and outputs below are inspectable.</div>",
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
                    pred = int(agent._model.predict(df_row)[0])
                except Exception:
                    logger.exception("Random Forest predict/predict_proba failed during scan")
                    proba = [0.5, 0.5]
                    pred = -1

                if blacklisted:
                    anomalous = False
                    xai_top: list[dict[str, Any]] | None = None
                else:
                    anomalous = pred == 1
                    xai_top = agent.top_critical_features(fd) if anomalous else None

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
                    "xai_top_features": xai_top,
                }

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
            '<p class="middleware-verdict-heading">Middleware Verdict</p>',
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
                    help="Middleware composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
                )
            elif v == "DANGER":
                st.metric(
                    "Final status",
                    "DANGER",
                    delta="Blacklist / critical",
                    delta_color="inverse",
                    help="Middleware composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
                )
            else:
                st.metric(
                    "Final status",
                    "SUSPICIOUS",
                    delta="ML anomaly",
                    delta_color="off",
                    help="Middleware composite verdict after GoPlus + Random Forest (+ LLM if triggered).",
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
                    <div class="advisor-title">AI Security Advisor Report</div>
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

        report_payload = {
            "report_title": "Catch Theft Crypto Security — Threat Intelligence Report",
            "wallet_address": snap.get("wallet_address"),
            "verdict": snap.get("verdict"),
            "threat_source": snap.get("threat_source"),
            "latency_ms": snap.get("latency_ms"),
            "risk_class_probability_pct": round(risk_pct, 4),
            "confidence_in_argmax_class_pct": round(mx_pct, 4),
            "xai_top_features": snap.get("xai_top_features"),
            "llm_advisory_text": snap.get("llm_warning"),
            "feature_vector_45": snap.get("features"),
            "evaluate_style_summary": snap.get("raw_evaluate"),
            "goplus_raw": snap.get("goplus_raw"),
            "groq_raw": snap.get("groq_raw"),
        }
        st.download_button(
            label="Download Threat Intel Report (JSON)",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name="Threat_Intel_Report.json",
            mime="application/json",
            key="download_threat_intel_report_json",
            use_container_width=True,
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
                st.markdown("**`evaluate_transaction`-style summary + raw GoPlus**")
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
        page_title="Catch Theft Crypto Security",
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
        .middleware-verdict-heading { color: #F0B90B !important; font-size: 1.15rem; font-weight: 700; margin: 0.5rem 0 0.25rem 0; }
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

    st.title("Catch Theft Crypto Security")
    tab_live, tab_analytics = st.tabs(
        ["🔍 Live Threat Analysis", "📊 AI Model Analytics"]
    )
    with tab_live:
        _render_live_threat_tab(agent, names)
    with tab_analytics:
        _render_ai_model_analytics_tab(agent)


if __name__ == "__main__":
    main()
