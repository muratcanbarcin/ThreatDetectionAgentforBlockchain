"""
Web3 Threat Detection — 45 boyutlu özellik vektörüyle uyumlu güvenlik paneli.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from middleware import ThreatDetectionAgent, _fraud_profile_for_demo

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "transaction_dataset.csv"

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


@st.cache_resource
def get_agent() -> ThreatDetectionAgent:
    return ThreatDetectionAgent()


@st.cache_data(show_spinner=False)
def _profile_from_dataset(flag: Literal[0, 1], feature_names: tuple[str, ...]) -> dict[str, float]:
    """Eğitim CSV’sinden FLAG=0 (güvenli) veya FLAG=1 (dolandırıcılık) için tam 45 özellik."""
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
    return f"fld_{i}"


def init_feature_session(names: list[str]) -> None:
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
    return {
        n: float(st.session_state.get(_field_session_key(i), 0.0) or 0.0)
        for i, n in enumerate(names)
    }


def push_profile_to_session(names: list[str], profile: dict[str, float]) -> None:
    for i, n in enumerate(names):
        st.session_state[_field_session_key(i)] = float(profile.get(n, 0.0) or 0.0)


def apply_scenario_safe(agent: ThreatDetectionAgent) -> None:
    names = agent._feature_names
    tup = tuple(names)
    prof = _profile_from_dataset(0, tup)
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    push_profile_to_session(names, prof)


def apply_scenario_known_threat(agent: ThreatDetectionAgent) -> None:
    names = agent._feature_names
    tup = tuple(names)
    prof = _profile_from_dataset(0, tup)
    st.session_state.ui_addr = KNOWN_THREAT_ADDRESS
    st.session_state.demo_force_blacklist = True
    push_profile_to_session(names, prof)


def apply_scenario_anomaly(agent: ThreatDetectionAgent) -> None:
    names = agent._feature_names
    try:
        prof = _fraud_profile_for_demo(names, agent._model)
    except FileNotFoundError:
        prof = _profile_from_dataset(1, tuple(names))
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    push_profile_to_session(names, prof)


def render_sidebar(agent: ThreatDetectionAgent) -> None:
    names = agent._feature_names
    init_feature_session(names)

    st.markdown("### 🧪 Demo Scenarios")
    st.caption(
        "Her senaryo **45 özelliğin tamamını** eğitim dağılımına yakın değerlerle doldurur. "
        "**Known Threat** kara listeyi sunum modunda simüle eder (GoPlus atlanır)."
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
    with st.expander("🛠️ Advanced Technical Features (41 More)", expanded=False):
        st.caption("Kalan özellikler — manuel stres testi için.")
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


def main() -> None:
    st.set_page_config(
        page_title="Web3 Threat Detection",
        page_icon="🛡️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .dash-hero {
            font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .dash-sub { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        .advisor-safe {
            border-radius: 12px; padding: 1.25rem; margin-top: 0.75rem;
            border: 1px solid rgba(34,197,94,0.45);
            background: linear-gradient(145deg, rgba(34,197,94,0.12), rgba(15,23,42,0.6));
        }
        .advisor-risk {
            border-radius: 12px; padding: 1.25rem; margin-top: 0.75rem;
            border: 1px solid rgba(248,113,113,0.5);
            background: linear-gradient(145deg, rgba(239,68,68,0.12), rgba(15,23,42,0.65));
        }
        .advisor-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem; }
        .advisor-body { font-size: 1.05rem; line-height: 1.55; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    agent = get_agent()
    names = agent._feature_names
    init_feature_session(names)

    with st.sidebar:
        render_sidebar(agent)

    st.markdown('<div class="dash-hero">🛡️ Web3 Threat Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-sub">45 boyutlu on-chain özellik vektörü → GoPlus → Random Forest '
        "(olasılık) → Groq. Kara kutu değil; tüm girdi/çıktı aşağıda denetlenebilir.</div>",
        unsafe_allow_html=True,
    )

    analyze = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        addr = (st.session_state.get("ui_addr") or "").strip()
        if not addr:
            st.warning("Lütfen bir cüzdan adresi girin.")
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
                    label="🔍 Querying Blockchain Intelligence APIs (GoPlus)...",
                    state="running",
                )
                time.sleep(0.6)
                if force_bl:
                    blacklisted = True
                    goplus_raw: dict[str, Any] = {
                        "_demo_simulation": True,
                        "message": "Known Threat senaryosu: GoPlus simüle kara liste.",
                    }
                else:
                    blacklisted, goplus_raw = agent.fetch_goplus_security(addr)

                status.update(
                    label="🧠 Processing 45-Dimensional Feature Vector...",
                    state="running",
                )
                time.sleep(0.6)

                status.update(
                    label="🛡️ Evaluating via Random Forest Engine...",
                    state="running",
                )
                time.sleep(0.6)

                df_row = pd.DataFrame([fd], columns=names)
                try:
                    proba = agent._model.predict_proba(df_row)[0].tolist()
                    pred = int(agent._model.predict(df_row)[0])
                except Exception:
                    proba = [0.5, 0.5]
                    pred = -1

                if blacklisted:
                    anomalous = False
                else:
                    anomalous = pred == 1

                threat = blacklisted or anomalous
                llm_text: str | None = None
                groq_raw: Any = None

                if threat:
                    status.update(
                        label="🤖 Generating Contextual Report (Groq LLM)...",
                        state="running",
                    )
                    time.sleep(0.6)
                    if blacklisted:
                        reason = (
                            "GoPlus kara liste / güvenlik bayrakları "
                            "(phishing, kötü amaçlı davranış veya hırsızlık riski)"
                        )
                    else:
                        reason = "makine öğrenmesi tabanlı anomali skoru"
                    detail = agent.generate_llm_warning_detailed(addr, reason)
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
                    "predict_proba": {"class_0_legit": proba[0], "class_1_risk": proba[1]}
                    if len(proba) > 1
                    else proba,
                    "confidence_in_argmax_class_pct": round(conf_pct, 2),
                    "risk_class_probability_pct": round(risk_pct, 2),
                }

                st.session_state["last_scan"] = {
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

        m1, m2, m3 = st.columns(3)
        with m1:
            if v == "SAFE":
                st.metric("Verdict", "SAFE", delta="İşlem imzalanabilir", delta_color="normal")
            elif v == "DANGER":
                st.metric("Verdict", "DANGER", delta="Kara liste / yüksek risk", delta_color="inverse")
            else:
                st.metric("Verdict", "SUSPICIOUS", delta="ML anomali", delta_color="off")
        with m2:
            st.metric(
                "Confidence Score",
                f"{mx_pct:.1f}%",
                help="Random Forest predict_proba: argmax sınıf olasılığı",
            )
            st.caption(f"Risk sınıfı (1) olasılığı: **{risk_pct:.1f}%**")
        with m3:
            st.metric("Processing Latency", f"{lat:,.0f} ms")

        st.divider()
        if v == "SAFE":
            st.success(
                "**No anomalies detected. Transaction is safe to sign.**\n\n"
                f"Model risk olasılığı: **{risk_pct:.1f}%** · Latency: **{lat:,.0f} ms**"
            )
        else:
            st.warning("**Risk pipeline tetiklendi** — aşağıdaki danışman raporunu okuyun.")
            llm = snap.get("llm_warning") or ""
            safe_llm = html.escape(str(llm))
            st.markdown(
                f"""
                <div class="advisor-risk">
                    <div class="advisor-title">🤖 AI Security Advisor Report</div>
                    <div class="advisor-body">{safe_llm}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("📋 Raw Model Input (45 Vector)", expanded=False):
        fd_show = st.session_state.get("feature_dict") or feature_dict_from_session(names)
        st.json(fd_show)

    with st.expander("🛠️ System Logs & Raw Payload", expanded=False):
        if not snap:
            st.caption("Önce **Analyze Transaction** çalıştırın.")
        else:
            cleft, cright = st.columns(2)
            with cleft:
                st.markdown("**Random Forest girdi vektörü (45)**")
                st.json(snap.get("features") or {})
            with cright:
                st.markdown("**`evaluate_transaction` tarzı özet + ham GoPlus**")
                st.json(
                    {
                        "evaluate_style_summary": snap.get("raw_evaluate"),
                        "goplus_raw": snap.get("goplus_raw"),
                        "groq_raw": snap.get("groq_raw"),
                    }
                )


if __name__ == "__main__":
    main()
