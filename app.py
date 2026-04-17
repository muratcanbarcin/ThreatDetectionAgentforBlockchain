"""
Streamlit güvenlik paneli: Hybrid Threat Detection Middleware.
"""

from __future__ import annotations

import html
import json
import time
from typing import Any

import pandas as pd
import streamlit as st

from middleware import ThreatDetectionAgent, _fraud_profile_for_demo

SIDEBAR_NUMERIC_FIELDS: list[tuple[str, str]] = [
    (
        "Total Transactions",
        "total transactions (including tnx to create contract",
    ),
    ("Total Ether Sent", "total Ether sent"),
    ("Total Ether Received", "total ether received"),
    ("Sent Transactions", "Sent tnx"),
]

SAFE_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
MALICIOUS_DEMO_ADDRESS = "0x0000000000000000000000000000000000000bad"
# RF=1 için 45 boyutlu tam vektör (yalnızca 4 sidebar alanı yetmez)
FEATURE_VECTOR_OVERRIDE_KEY = "feature_vector_45"


@st.cache_resource
def get_agent() -> ThreatDetectionAgent:
    return ThreatDetectionAgent()


def _init_session() -> None:
    if "ui_addr" not in st.session_state:
        st.session_state.ui_addr = SAFE_ADDRESS
    if "demo_force_blacklist" not in st.session_state:
        st.session_state.demo_force_blacklist = False
    for _lb, col in SIDEBAR_NUMERIC_FIELDS:
        key = f"ui_feat_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0


def _feat_keys() -> list[str]:
    return [col for _lb, col in SIDEBAR_NUMERIC_FIELDS]


def _apply_safe_demo() -> None:
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    st.session_state.pop(FEATURE_VECTOR_OVERRIDE_KEY, None)
    for col in _feat_keys():
        st.session_state[f"ui_feat_{col}"] = 0.0


def _apply_malicious_demo() -> None:
    st.session_state.ui_addr = MALICIOUS_DEMO_ADDRESS
    st.session_state.demo_force_blacklist = True
    st.session_state.pop(FEATURE_VECTOR_OVERRIDE_KEY, None)
    for col in _feat_keys():
        st.session_state[f"ui_feat_{col}"] = 0.0


def _apply_anomaly_demo(agent: ThreatDetectionAgent) -> None:
    st.session_state.ui_addr = SAFE_ADDRESS
    st.session_state.demo_force_blacklist = False
    try:
        profile = _fraud_profile_for_demo(agent._feature_names, agent._model)
        st.session_state[FEATURE_VECTOR_OVERRIDE_KEY] = {
            k: float(profile[k]) for k in agent._feature_names
        }
        for col in _feat_keys():
            st.session_state[f"ui_feat_{col}"] = float(profile.get(col, 0.0))
    except FileNotFoundError:
        st.session_state.pop(FEATURE_VECTOR_OVERRIDE_KEY, None)
        for col in _feat_keys():
            st.session_state[f"ui_feat_{col}"] = 0.0
        st.session_state[f"ui_feat_{_feat_keys()[0]}"] = 1_000_000.0
        st.session_state[f"ui_feat_{_feat_keys()[3]}"] = 500_000.0


def build_features_dict(agent: ThreatDetectionAgent) -> dict[str, float]:
    names = agent._feature_names
    override = st.session_state.get(FEATURE_VECTOR_OVERRIDE_KEY)
    if isinstance(override, dict) and all(k in override for k in names):
        out = {k: float(override.get(k, 0) or 0) for k in names}
        for _label, col in SIDEBAR_NUMERIC_FIELDS:
            if col in out:
                key = f"ui_feat_{col}"
                out[col] = float(st.session_state.get(key, out[col]) or 0)
        return out

    out = {name: 0.0 for name in names}
    for _label, col in SIDEBAR_NUMERIC_FIELDS:
        if col in out:
            key = f"ui_feat_{col}"
            out[col] = float(st.session_state.get(key, 0.0) or 0.0)
    return out


def main() -> None:
    st.set_page_config(
        page_title="Web3 Threat Detection",
        page_icon="🛡️",
        layout="wide",
    )
    _init_session()

    st.markdown(
        """
        <style>
        .hero-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
        .hero-sub {
            color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.25rem;
        }
        .panel-safe {
            background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(0,0,0,0.25));
            border: 2px solid #10b981; border-radius: 14px; padding: 1.5rem; margin: 0.5rem 0;
        }
        .panel-deny {
            background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(0,0,0,0.3));
            border: 2px solid #ef4444; border-radius: 14px; padding: 1.5rem; margin: 0.5rem 0;
        }
        .tx-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 10px; padding: 0.85rem 1rem; margin: 0.35rem 0;
        }
        .mono { font-family: ui-monospace, monospace; font-size: 0.82rem; word-break: break-all; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="hero-title">🛡️ Web3 Threat Detection Middleware</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Bu panel, arka plandaki middleware’in sırasıyla '
        "<strong>GoPlus</strong> (kara liste), <strong>Random Forest</strong> (anomali) ve "
        "<strong>Groq</strong> (bağlamsal uyarı) katmanlarını nasıl kullandığını adım adım gösterir.</p>",
        unsafe_allow_html=True,
    )

    agent = get_agent()

    with st.sidebar:
        st.markdown("### 🧪 Quick Demo Scenarios")
        st.caption(
            "Sunum için hazır senaryolar: form alanları otomatik dolar. "
            "“Malicious” senaryosu kara listeyi **sunum modunda** simüle eder (GoPlus yerine). "
            "“Anomaly” eğitim verisinden **45 özelliklik tam vektör** yükler (yalnızca 4 alan RF için yetersiz)."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Safe Address", use_container_width=True):
                _apply_safe_demo()
                st.session_state.pop("last_scan", None)
                st.rerun()
        with c2:
            if st.button("Load Malicious Address", use_container_width=True):
                _apply_malicious_demo()
                st.session_state.pop("last_scan", None)
                st.rerun()
        if st.button("Load Anomaly Behavior", use_container_width=True):
            _apply_anomaly_demo(agent)
            st.session_state.pop("last_scan", None)
            st.rerun()

        st.divider()
        st.markdown("### Transaction inputs")
        st.text_input("Cüzdan adresi", key="ui_addr", help="Ethereum mainnet (chain_id=1)")
        st.caption("Öne çıkan 4 özellik; kalan ~41 sütun 0 kabul edilir.")
        for label, col in SIDEBAR_NUMERIC_FIELDS:
            st.number_input(
                label,
                min_value=0.0,
                step=1.0,
                format="%.6f",
                key=f"ui_feat_{col}",
            )

    scan = st.button(
        "🔍 Scan Transaction",
        type="primary",
        use_container_width=True,
        help="GoPlus → Random Forest → (gerekirse) Groq sırasıyla çalışır",
    )

    if scan:
        addr = (st.session_state.get("ui_addr") or "").strip()
        if not addr:
            st.warning("Lütfen bir cüzdan adresi girin.")
        else:
            features_dict = build_features_dict(agent)
            force_bl = bool(st.session_state.get("demo_force_blacklist"))
            t0 = time.perf_counter()
            llm_text: str | None = None

            uses_full_profile = isinstance(
                st.session_state.get(FEATURE_VECTOR_OVERRIDE_KEY), dict
            )
            telemetry: dict[str, Any] = {
                "address": addr,
                "features_to_model": features_dict,
                "demo_force_blacklist": force_bl,
                "feature_input_mode": (
                    "training_fraud_profile_45d"
                    if uses_full_profile
                    else "sidebar_four_plus_zeros"
                ),
            }

            with st.status("Middleware pipeline", expanded=True) as status:
                line1 = status.empty()
                line1.markdown("⏳ **1.** Checking global blacklists (GoPlus API)...")
                if force_bl:
                    blacklisted = True
                    goplus_raw: dict[str, Any] = {
                        "_demo_simulation": True,
                        "message": "Sunum: kara liste senaryosu — gerçek GoPlus çağrısı atlandı.",
                    }
                else:
                    blacklisted, goplus_raw = agent.fetch_goplus_security(addr)
                telemetry["goplus_response"] = goplus_raw
                line1.markdown("✅ **1.** Checking global blacklists (GoPlus API)...")

                line2 = status.empty()
                line2.markdown("⏳ **2.** Running transaction through Random Forest AI...")
                if blacklisted:
                    anomalous = False
                    rf_note = "Atlandı (kara liste önceliği)"
                else:
                    anomalous = agent.check_anomaly(features_dict)
                    rf_note = "Tamamlandı"
                try:
                    df_row = pd.DataFrame(
                        [features_dict], columns=agent._feature_names
                    )
                    pred = int(agent._model.predict(df_row)[0])
                except Exception:
                    pred = -1
                telemetry["random_forest"] = {
                    "prediction_class": pred,
                    "anomaly_flag": anomalous,
                    "note": rf_note,
                }
                line2.markdown("✅ **2.** Running transaction through Random Forest AI...")

                threat = blacklisted or anomalous
                line3 = status.empty()
                groq_raw: Any = None
                if threat:
                    line3.markdown(
                        "⏳ **3.** Generating contextual threat intelligence (Groq LLM)..."
                    )
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
                    telemetry["groq_response"] = groq_raw
                    telemetry["llm_used"] = bool(detail.get("used_llm"))
                    line3.markdown(
                        "✅ **3.** Generating contextual threat intelligence (Groq LLM)..."
                    )
                else:
                    line3.markdown(
                        "⏭️ **3.** Groq LLM atlandı — tehdit yok, **latency optimizasyonu** (LLM bypass)"
                    )
                    telemetry["groq_response"] = None
                    telemetry["llm_used"] = False

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if threat:
                status_str = "Denied/Pending"
            else:
                status_str = "Allow"

            st.session_state.demo_force_blacklist = False

            st.session_state["last_scan"] = {
                "status": status_str,
                "llm_warning": llm_text,
                "latency_ms": round(latency_ms, 2),
                "telemetry": telemetry,
                "blacklisted": blacklisted,
                "anomalous": anomalous,
            }

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Transaction Details")
        st.markdown("Kullanıcı girdilerinin özeti — middleware’e giden işlem bağlamı.")
        addr = (st.session_state.get("ui_addr") or "").strip()
        st.markdown(
            f'<div class="tx-card"><strong>Adres</strong><div class="mono">{html.escape(addr) or "—"}</div></div>',
            unsafe_allow_html=True,
        )
        fv = build_features_dict(agent)
        mcols = st.columns(2)
        for i, (_lb, col) in enumerate(SIDEBAR_NUMERIC_FIELDS):
            with mcols[i % 2]:
                st.metric(label=_lb, value=f"{fv[col]:,.4f}")
        with st.expander("Tüm özellik vektörü (45 boyut)", expanded=False):
            st.json(fv)

    with col_right:
        st.subheader("Middleware Live Analysis")
        snap = st.session_state.get("last_scan")
        if snap:
            status = snap["status"]
            lat = snap["latency_ms"]
            llm = snap.get("llm_warning")

            if status == "Allow":
                st.markdown(
                    f"""
                    <div class="panel-safe">
                        <div style="font-size:1.35rem;font-weight:700;color:#34d399;margin-bottom:0.5rem;">
                            İşlem Onaylandı
                        </div>
                        <div style="font-size:1.05rem;">
                            Bu adres ve işlem özellikleri için tehdit sinyali yok.
                            Middleware <strong>GoPlus</strong> ve <strong>Random Forest</strong>
                            katmanlarından geçti; LLM çağrılmadı.
                        </div>
                        <div style="margin-top:1rem;font-size:1.1rem;font-weight:600;">
                            Gecikme (Latency): ~{lat:.1f} ms (LLM Bypassed for Optimization)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="panel-deny">
                        <div style="font-size:1.35rem;font-weight:700;color:#fca5a5;margin-bottom:0.35rem;">
                            İşlem Reddedildi / İncelemede
                        </div>
                        <div style="font-size:1rem;">
                            Kara liste veya ML anomali tespit edildi. Middleware, bağlam için
                            <strong>Groq LLM</strong> katmanını devreye aldı (veya anahtar yoksa yerel uyarı üretti).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if llm:
                    st.info("🛡️ **AI Security Advisor**\n\n" + str(llm))

            st.metric("Processing Latency (toplam)", f"{lat:,.2f} ms")
        else:
            st.info(
                "Sol tarafta adres ve özellikleri girin veya **Quick Demo** ile yükleyin; "
                "ardından **Scan Transaction** ile GoPlus → Random Forest → (isteğe bağlı) Groq "
                "zincirini çalıştırın."
            )

    with st.expander("Technical Details & Raw Data", expanded=False):
        st.markdown(
            "Aşağıda modele giden **tam özellik sözlüğü** ve **GoPlus / Groq ham yanıtları** "
            "(varsa) yer alır. Jüri sorularında doğrudan bu JSON’u gösterebilirsiniz."
        )
        snap = st.session_state.get("last_scan")
        if not snap:
            st.caption("Henüz tarama yapılmadı — Scan Transaction ile veri üretin.")
        else:
            tel = snap.get("telemetry") or {}
            st.caption(
                f"Özellik kaynağı: `{tel.get('feature_input_mode', '—')}` "
                "(Anomaly demo = eğitimden 45 boyutlu vektör)"
            )
            st.markdown("**`features_to_model` (45 özellik)**")
            st.code(
                json.dumps(tel.get("features_to_model", {}), indent=2, ensure_ascii=False),
                language="json",
            )
            st.markdown("**`goplus_response` (GoPlus API)**")
            st.code(
                json.dumps(tel.get("goplus_response", {}), indent=2, ensure_ascii=False),
                language="json",
            )
            st.markdown("**`random_forest` (özet)**")
            st.code(
                json.dumps(tel.get("random_forest", {}), indent=2, ensure_ascii=False),
                language="json",
            )
            st.markdown("**`groq_response` (Groq tam yanıt veya null)**")
            gr = tel.get("groq_response")
            st.code(
                json.dumps(gr, indent=2, ensure_ascii=False) if gr is not None else "null",
                language="json",
            )


if __name__ == "__main__":
    main()
